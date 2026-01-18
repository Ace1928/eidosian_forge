import numpy as np
from numpy.core.multiarray import normalize_axis_index
from scipy._lib._util import _nan_allsame, _contains_nan
from ._stats_py import _chk_asarray
def _nanvariation(a, *, axis=0, ddof=0, keepdims=False):
    """
    Private version of `variation` that ignores nan.

    `a` must be a numpy array.
    `axis` is assumed to be normalized, i.e. 0 <= axis < a.ndim.
    """
    a_isnan = np.isnan(a)
    all_nan = a_isnan.all(axis=axis, keepdims=True)
    all_nan_full = np.broadcast_to(all_nan, a.shape)
    all_zero = (a_isnan | (a == 0)).all(axis=axis, keepdims=True) & ~all_nan
    ngood = a.shape[axis] - np.expand_dims(np.count_nonzero(a_isnan, axis=axis), axis)
    ddof_too_big = ddof > ngood
    ddof_equal_n = ddof == ngood
    is_const = _nan_allsame(a, axis=axis, keepdims=True)
    a2 = a.copy()
    a2[all_nan_full] = 1.0
    mean_a = np.nanmean(a2, axis=axis, keepdims=True)
    a2[np.broadcast_to(ddof_too_big, a2.shape) | ddof_equal_n] = 1.0
    with np.errstate(invalid='ignore'):
        std_a = np.nanstd(a2, axis=axis, ddof=ddof, keepdims=True)
    del a2
    sum_zero = np.nansum(a, axis=axis, keepdims=True) == 0
    mean_a[sum_zero] = 1.0
    result = std_a / mean_a
    result[~is_const & sum_zero] = np.inf
    signed_inf_mask = ~is_const & ddof_equal_n
    result[signed_inf_mask] = np.sign(mean_a[signed_inf_mask]) * np.inf
    nan_mask = all_zero | all_nan | ddof_too_big | ddof_equal_n & is_const
    result[nan_mask] = np.nan
    if not keepdims:
        result = np.squeeze(result, axis=axis)
        if result.shape == ():
            result = result[()]
    return result