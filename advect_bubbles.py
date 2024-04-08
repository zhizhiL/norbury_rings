import numpy as np
import scipy as sp
import pickle
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import matplotlib.pyplot as plt

R, gravity, Fr = 1.99, True, 1

x_grid = np.load('velocity_results/alpha04_2D_x.npy')
y_grid = np.load('velocity_results/alpha04_2D_y.npy')
Ux = np.load('velocity_results/alpha04_2D_Ux.npy')
Uy = np.load('velocity_results/alpha04_2D_Uy.npy')
dUxdx = np.load('velocity_results/alpha04_2D_dUxdx.npy')
dUxdy = np.load('velocity_results/alpha04_2D_dUxdy.npy')
dUydx = np.load('velocity_results/alpha04_2D_dUydx.npy')
dUydy = np.load('velocity_results/alpha04_2D_dUydy.npy')
geometry = np.load('velocity_results/alpha04_2D_geometry.npy', allow_pickle=True)
x_core, y_core, y_core_lower, x_ring, y_ring = geometry.T

# load interpolation functions
with open('velocity_results/interp_functions_alpha04_2D.pkl', 'rb') as f:
    interp_Ux, interp_Uy, interp_dUxdx, interp_dUxdy, interp_dUydx, interp_dUydy = pickle.load(f)



def solve_ivp_active(args):

    def active_tracer_traj(t, Z):
        xp, yp, dxpdt, dypdt = Z

        Uxp = interp_Ux(xp, yp)[0]
        Uyp = interp_Uy(xp, yp)[0]
        dUxdx_p = interp_dUxdx(xp, yp)[0]
        dUxdy_p = interp_dUxdy(xp, yp)[0]
        dUydx_p = interp_dUydx(xp, yp)[0]
        dUydy_p = interp_dUydy(xp, yp)[0]

        dUxdt = 0
        dUydt = 0

        ddxpdtt = R*(Uxp - dxpdt)/St + (3*R/2) * (dUxdt + Uxp*dUxdx_p + Uyp*dUxdy_p)
        ddypdtt = R*(Uyp - dypdt)/St + (3*R/2) * (dUydt + Uxp*dUydx_p + Uyp*dUydy_p)  - gravity * (1-3*R/2) / (Fr**2)

        return [dxpdt, dypdt, ddxpdtt, ddypdtt]
    
    q0, t_span = args
    x0, y0, vx0, vy0, St = q0
    sol = sp.integrate.solve_ivp(active_tracer_traj, [t_span[0], t_span[-1]], [x0, y0, vx0, vy0], method='RK45', t_eval=t_span)

    return sol.y[0], sol.y[1], sol.y[2], sol.y[3]

def advect_bubbles(bubbles_df_to_advect, t0, tf, plot_path = False, this_ax = None, color=None):
    initial_states = bubbles_df_to_advect[:, 1:6]
    t_span = np.linspace(t0, tf, 1000)

    n_proc = 12

    with Pool(n_proc) as pool:
        args = list(zip(initial_states, [t_span]*len(initial_states)))
        res = pool.map(solve_ivp_active, args)

    res_array = np.stack(res, axis=0) # shape (N_bubbles, 4, len(t_span))

    if plot_path:
        plt.sca(ax=this_ax)
        plt.scatter(res_array[:, 0, :].T, res_array[:, 1, :].T, color=color, s=0.01, linewidths=0)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.show()
    
    return res_array[:, :, -1]