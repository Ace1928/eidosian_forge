from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def _assert_matching_namespace(actual, desired):
    actual = actual if isinstance(actual, tuple) else (actual,)
    desired_space = array_namespace(desired)
    for arr in actual:
        arr_space = array_namespace(arr)
        assert_(arr_space == desired_space, f'Namespaces do not match.\nActual: {arr_space.__name__}\nDesired: {desired_space.__name__}')