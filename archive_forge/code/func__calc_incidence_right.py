import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def _calc_incidence_right(time, status, weights=None):
    """
    Calculate the cumulative incidence function and its standard error.
    """
    status0 = (status >= 1).astype(np.float64)
    sp, utime, rtime, n, d = _calc_survfunc_right(time, status0, weights, compress=False, retall=False)
    ngrp = int(status.max())
    d = []
    for k in range(ngrp):
        status0 = (status == k + 1).astype(np.float64)
        if weights is None:
            d0 = np.bincount(rtime, weights=status0, minlength=len(utime))
        else:
            d0 = np.bincount(rtime, weights=status0 * weights, minlength=len(utime))
        d.append(d0)
    ip = []
    sp0 = np.r_[1, sp[:-1]] / n
    for k in range(ngrp):
        ip0 = np.cumsum(sp0 * d[k])
        ip.append(ip0)
    if weights is not None:
        return (ip, None, utime)
    se = []
    da = sum(d)
    for k in range(ngrp):
        ra = da / (n * (n - da))
        v = ip[k] ** 2 * np.cumsum(ra)
        v -= 2 * ip[k] * np.cumsum(ip[k] * ra)
        v += np.cumsum(ip[k] ** 2 * ra)
        ra = (n - d[k]) * d[k] / n
        v += np.cumsum(sp0 ** 2 * ra)
        ra = sp0 * d[k] / n
        v -= 2 * ip[k] * np.cumsum(ra)
        v += 2 * np.cumsum(ip[k] * ra)
        se.append(np.sqrt(v))
    return (ip, se, utime)