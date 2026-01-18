import warnings
import numpy as np
from scipy import interpolate, stats
def _rankdata_no_ties(x):
    """rankdata without ties for 2-d array

    This is a simplified version for ranking data if there are no ties.
    Works vectorized across columns.

    See Also
    --------
    scipy.stats.rankdata

    """
    nobs, k_vars = x.shape
    ranks = np.ones((nobs, k_vars))
    sidx = np.argsort(x, axis=0)
    ranks[sidx, np.arange(k_vars)] = np.arange(1, nobs + 1)[:, None]
    return ranks