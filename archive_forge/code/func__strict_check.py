from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def _strict_check(actual, desired, xp, check_namespace=True, check_dtype=True, check_shape=True):
    if check_namespace:
        _assert_matching_namespace(actual, desired)
    desired = xp.asarray(desired)
    if check_dtype:
        assert_(actual.dtype == desired.dtype, f'dtypes do not match.\nActual: {actual.dtype}\nDesired: {desired.dtype}')
    if check_shape:
        assert_(actual.shape == desired.shape, f'Shapes do not match.\nActual: {actual.shape}\nDesired: {desired.shape}')
        _check_scalar(actual, desired, xp)
    desired = xp.broadcast_to(desired, actual.shape)
    return desired