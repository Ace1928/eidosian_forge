from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def _fitstart_beta(self, x, fixed=None):
    """method of moment estimator as starting values for beta distribution

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated
    fixed : None or array_like
        sequence of numbers and np.nan to indicate fixed parameters and parameters
        to estimate

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    References
    ----------
    for method of moment estimator for known loc and scale
    https://en.wikipedia.org/wiki/Beta_distribution#Parameter_estimation
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
    NIST reference also includes reference to MLE in
    Johnson, Kotz, and Balakrishan, Volume II, pages 221-235

    """
    a, b = (x.min(), x.max())
    eps = (a - b) * 0.01
    if fixed is None:
        loc = a - eps
        scale = (a - b) * (1 + 2 * eps)
    else:
        if np.isnan(fixed[-2]):
            loc = a - eps
        else:
            loc = fixed[-2]
        if np.isnan(fixed[-1]):
            scale = b + eps - loc
        else:
            scale = fixed[-1]
    scale = float(scale)
    xtrans = (x - loc) / scale
    xm = xtrans.mean()
    xv = xtrans.var()
    tmp = xm * (1 - xm) / xv - 1
    p = xm * tmp
    q = (1 - xm) * tmp
    return (p, q, loc, scale)