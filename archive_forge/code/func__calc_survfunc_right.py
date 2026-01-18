import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def _calc_survfunc_right(time, status, weights=None, entry=None, compress=True, retall=True):
    """
    Calculate the survival function and its standard error for a single
    group.
    """
    if entry is None:
        utime, rtime = np.unique(time, return_inverse=True)
    else:
        tx = np.concatenate((time, entry))
        utime, rtime = np.unique(tx, return_inverse=True)
        rtime = rtime[0:len(time)]
    ml = len(utime)
    if weights is None:
        d = np.bincount(rtime, weights=status, minlength=ml)
    else:
        d = np.bincount(rtime, weights=status * weights, minlength=ml)
    if weights is None:
        n = np.bincount(rtime, minlength=ml)
    else:
        n = np.bincount(rtime, weights=weights, minlength=ml)
    if entry is not None:
        n = np.cumsum(n) - n
        rentry = np.searchsorted(utime, entry, side='left')
        if weights is None:
            n0 = np.bincount(rentry, minlength=ml)
        else:
            n0 = np.bincount(rentry, weights=weights, minlength=ml)
        n0 = np.cumsum(n0) - n0
        n = n0 - n
    else:
        n = np.cumsum(n[::-1])[::-1]
    if compress:
        ii = np.flatnonzero(d > 0)
        d = d[ii]
        n = n[ii]
        utime = utime[ii]
    sp = 1 - d / n.astype(np.float64)
    ii = sp < 1e-16
    sp[ii] = 1e-16
    sp = np.log(sp)
    sp = np.cumsum(sp)
    sp = np.exp(sp)
    sp[ii] = 0
    if not retall:
        return (sp, utime, rtime, n, d)
    if weights is None:
        denom = n * (n - d)
        denom = np.clip(denom, 1e-12, np.inf)
        se = d / denom.astype(np.float64)
        se[(n == d) | (n == 0)] = np.nan
        se = np.cumsum(se)
        se = np.sqrt(se)
        locs = np.isfinite(se) | (sp != 0)
        se[locs] *= sp[locs]
        se[~locs] = np.nan
    else:
        se = d / (n * n).astype(np.float64)
        se = np.cumsum(se)
        se = np.sqrt(se)
    return (sp, se, utime, rtime, n, d)