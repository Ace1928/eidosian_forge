from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _normalize_split(proportion):
    """
    return a list of proportions of the available space given the division
    if only a number is given, it will assume a split in two pieces
    """
    if not iterable(proportion):
        if proportion == 0:
            proportion = array([0.0, 1.0])
        elif proportion >= 1:
            proportion = array([1.0, 0.0])
        elif proportion < 0:
            raise ValueError('proportions should be positive,given value: {}'.format(proportion))
        else:
            proportion = array([proportion, 1.0 - proportion])
    proportion = np.asarray(proportion, dtype=float)
    if np.any(proportion < 0):
        raise ValueError('proportions should be positive,given value: {}'.format(proportion))
    if np.allclose(proportion, 0):
        raise ValueError('at least one proportion should be greater than zerogiven value: {}'.format(proportion))
    if len(proportion) < 2:
        return array([0.0, 1.0])
    left = r_[0, cumsum(proportion)]
    left /= left[-1] * 1.0
    return left