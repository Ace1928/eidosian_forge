from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def _fitstart_poisson(self, x, fixed=None):
    """maximum likelihood estimator as starting values for Poisson distribution

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
    MLE :
    https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood

    """
    a = x.min()
    eps = 0
    if fixed is None:
        loc = a - eps
    elif np.isnan(fixed[-1]):
        loc = a - eps
    else:
        loc = fixed[-1]
    xtrans = x - loc
    lambd = xtrans.mean()
    return (lambd, loc)