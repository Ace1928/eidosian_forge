import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    """ Input validation and standardization for mannwhitneyu """
    x, y = (np.atleast_1d(x), np.atleast_1d(y))
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')
    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')
    alternatives = {'two-sided', 'less', 'greater'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    methods = {'asymptotic', 'exact', 'auto'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f'`method` must be one of {methods}.')
    return (x, y, use_continuity, alternative, axis_int, method)