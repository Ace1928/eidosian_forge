import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _euler_maclaurin_sum(fj, work):
    xr0, fr0, wr0 = (work.xr0, work.fr0, work.wr0)
    xl0, fl0, wl0 = (work.xl0, work.fl0, work.wl0)
    xj, fj, wj = (work.xj.T, fj.T, work.wj.T)
    n_x, n_active = xj.shape
    xr, xl = xj.reshape(2, n_x // 2, n_active).copy()
    fr, fl = fj.reshape(2, n_x // 2, n_active)
    wr, wl = wj.reshape(2, n_x // 2, n_active)
    invalid_r = ~np.isfinite(fr) | (wr == 0)
    invalid_l = ~np.isfinite(fl) | (wl == 0)
    xr[invalid_r] = -np.inf
    ir = np.argmax(xr, axis=0, keepdims=True)
    xr_max = np.take_along_axis(xr, ir, axis=0)[0]
    fr_max = np.take_along_axis(fr, ir, axis=0)[0]
    wr_max = np.take_along_axis(wr, ir, axis=0)[0]
    j = xr_max > xr0
    xr0[j] = xr_max[j]
    fr0[j] = fr_max[j]
    wr0[j] = wr_max[j]
    xl[invalid_l] = np.inf
    il = np.argmin(xl, axis=0, keepdims=True)
    xl_min = np.take_along_axis(xl, il, axis=0)[0]
    fl_min = np.take_along_axis(fl, il, axis=0)[0]
    wl_min = np.take_along_axis(wl, il, axis=0)[0]
    j = xl_min < xl0
    xl0[j] = xl_min[j]
    fl0[j] = fl_min[j]
    wl0[j] = wl_min[j]
    fj = fj.T
    flwl0 = fl0 + np.log(wl0) if work.log else fl0 * wl0
    frwr0 = fr0 + np.log(wr0) if work.log else fr0 * wr0
    magnitude = np.real if work.log else np.abs
    work.d4 = np.maximum(magnitude(flwl0), magnitude(frwr0))
    fr0b = np.broadcast_to(fr0[np.newaxis, :], fr.shape)
    fl0b = np.broadcast_to(fl0[np.newaxis, :], fl.shape)
    fr[invalid_r] = fr0b[invalid_r]
    fl[invalid_l] = fl0b[invalid_l]
    fjwj = fj + np.log(work.wj) if work.log else fj * work.wj
    Sn = special.logsumexp(fjwj + np.log(work.h), axis=-1) if work.log else np.sum(fjwj, axis=-1) * work.h
    work.xr0, work.fr0, work.wr0 = (xr0, fr0, wr0)
    work.xl0, work.fl0, work.wl0 = (xl0, fl0, wl0)
    return (fjwj, Sn)