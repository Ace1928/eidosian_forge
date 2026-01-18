import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _weight_checked(fn, n_args=2, default_axis=None, key=lambda x: x, weight_arg='w', squeeze=True, silent=False, ones_test=True, const_test=True, dup_test=True, split_test=True, dud_test=True, ma_safe=False, ma_very_safe=False, nan_safe=False, split_per=1.0, seed=0, compare_assert=partial(assert_allclose, atol=1e-05)):
    """runs fn on its arguments 2 or 3 ways, checks that the results are the same,
       then returns the same thing it would have returned before"""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        arrays = args[:n_args]
        rest = args[n_args:]
        weights = kwargs.get(weight_arg, None)
        axis = kwargs.get('axis', default_axis)
        chked = _chk_weights(arrays, weights=weights, axis=axis, force_weights=True, mask_screen=True)
        arrays, weights, axis = (chked[:-2], chked[-2], chked[-1])
        if squeeze:
            arrays = [np.atleast_1d(a.squeeze()) for a in arrays]
        try:
            args = tuple(arrays) + rest
            if ones_test:
                kwargs[weight_arg] = weights
                _rough_check(result, fn(*args, **kwargs), key=key)
            if const_test:
                kwargs[weight_arg] = weights * 101.0
                _rough_check(result, fn(*args, **kwargs), key=key)
                kwargs[weight_arg] = weights * 0.101
                try:
                    _rough_check(result, fn(*args, **kwargs), key=key)
                except Exception as e:
                    raise type(e)((e, arrays, weights)) from e
            if dud_test:
                dud_arrays, dud_weights = _rand_split(arrays, weights, axis, split_per=split_per, seed=seed)
                dud_weights[:weights.size] = weights
                dud_weights[weights.size:] = 0
                dud_args = tuple(dud_arrays) + rest
                kwargs[weight_arg] = dud_weights
                _rough_check(result, fn(*dud_args, **kwargs), key=key)
                for a in dud_arrays:
                    indexer = [slice(None)] * a.ndim
                    indexer[axis] = slice(weights.size, None)
                    indexer = tuple(indexer)
                    a[indexer] = a[indexer] * 101
                dud_args = tuple(dud_arrays) + rest
                _rough_check(result, fn(*dud_args, **kwargs), key=key)
                for a in dud_arrays:
                    indexer = [slice(None)] * a.ndim
                    indexer[axis] = slice(weights.size, None)
                    indexer = tuple(indexer)
                    a[indexer] = a[indexer] * np.nan
                if kwargs.get('nan_policy', None) == 'omit' and nan_safe:
                    dud_args = tuple(dud_arrays) + rest
                    _rough_check(result, fn(*dud_args, **kwargs), key=key)
                if ma_safe:
                    dud_arrays = [np.ma.masked_invalid(a) for a in dud_arrays]
                    dud_args = tuple(dud_arrays) + rest
                    _rough_check(result, fn(*dud_args, **kwargs), key=key)
                    if ma_very_safe:
                        kwargs[weight_arg] = None
                        _rough_check(result, fn(*dud_args, **kwargs), key=key)
                del dud_arrays, dud_args, dud_weights
            if dup_test:
                dup_arrays = [np.append(a, a, axis=axis) for a in arrays]
                dup_weights = np.append(weights, weights) / 2.0
                dup_args = tuple(dup_arrays) + rest
                kwargs[weight_arg] = dup_weights
                _rough_check(result, fn(*dup_args, **kwargs), key=key)
                del dup_args, dup_arrays, dup_weights
            if split_test and split_per > 0:
                split = _rand_split(arrays, weights, axis, split_per=split_per, seed=seed)
                split_arrays, split_weights = split
                split_args = tuple(split_arrays) + rest
                kwargs[weight_arg] = split_weights
                _rough_check(result, fn(*split_args, **kwargs), key=key)
        except NotImplementedError as e:
            if not silent:
                warnings.warn(f'{fn.__name__} NotImplemented weights: {e}', stacklevel=3)
        return result
    return wrapped