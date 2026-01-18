import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _tie_check(xy):
    """Find any ties in data"""
    _, t = np.unique(xy, return_counts=True, axis=-1)
    return np.any(t != 1)