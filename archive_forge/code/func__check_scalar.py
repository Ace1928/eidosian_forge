from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def _check_scalar(actual, desired, xp):
    if desired.shape != () or not is_numpy(xp):
        return
    desired = desired[()]
    assert_(xp.isscalar(actual) and xp.isscalar(desired) or (not xp.isscalar(actual) and (not xp.isscalar(desired))), f'Types do not match:\nActual: {type(actual)}\nDesired: {type(desired)}')