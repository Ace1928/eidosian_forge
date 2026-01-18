import numpy as np
from functools import partial
from scipy import stats
def _bws_input_validation(x, y, alternative, method):
    """ Input validation and standardization for bws test"""
    x, y = np.atleast_1d(x, y)
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError('`x` and `y` must be exactly one-dimensional.')
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')
    z = stats.rankdata(np.concatenate((x, y)))
    x, y = (z[:len(x)], z[len(x):])
    alternatives = {'two-sided', 'less', 'greater'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')
    method = stats.PermutationMethod() if method is None else method
    if not isinstance(method, stats.PermutationMethod):
        raise ValueError('`method` must be an instance of `scipy.stats.PermutationMethod`')
    return (x, y, alternative, method)