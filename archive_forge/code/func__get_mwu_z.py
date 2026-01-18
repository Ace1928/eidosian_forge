import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _get_mwu_z(U, n1, n2, ranks, axis=0, continuity=True):
    """Standardized MWU statistic"""
    mu = n1 * n2 / 2
    n = n1 + n2
    tie_term = np.apply_along_axis(_tie_term, -1, ranks)
    s = np.sqrt(n1 * n2 / 12 * (n + 1 - tie_term / (n * (n - 1))))
    numerator = U - mu
    if continuity:
        numerator -= 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z