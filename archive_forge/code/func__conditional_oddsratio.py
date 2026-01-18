import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _conditional_oddsratio(table):
    """
    Conditional MLE of the odds ratio for the 2x2 contingency table.
    """
    x, M, n, N = _hypergeom_params_from_table(table)
    lo, hi = nchypergeom_fisher.support(M, n, N, 1)
    if x == lo:
        return 0
    if x == hi:
        return np.inf
    nc = _nc_hypergeom_mean_inverse(x, M, n, N)
    return nc