import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def _integrate_rk12(x0, y0, dmap, f, maxlength, broken_streamlines=True):
    """
    2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as RK4 and RK45 solvers (using
    similar Python implementations) in most setups.
    """
    maxerror = 0.003
    maxds = min(1.0 / dmap.mask.nx, 1.0 / dmap.mask.ny, 0.1)
    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xyf_traj = []
    while True:
        try:
            if dmap.grid.within_grid(xi, yi):
                xyf_traj.append((xi, yi))
            else:
                raise OutOfBounds
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x, yi + ds * k1y)
        except OutOfBounds:
            if xyf_traj:
                ds, xyf_traj = _euler_step(xyf_traj, dmap, f)
                stotal += ds
            break
        except TerminateTrajectory:
            break
        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)
        ny, nx = dmap.grid.shape
        error = np.hypot((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1))
        if error < maxerror:
            xi += dx2
            yi += dy2
            try:
                dmap.update_trajectory(xi, yi, broken_streamlines)
            except InvalidIndexError:
                break
            if stotal + ds > maxlength:
                break
            stotal += ds
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)
    return (stotal, xyf_traj)