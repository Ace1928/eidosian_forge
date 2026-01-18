import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def _check_ninf_nan(dummy):
    msgform = 'csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)'
    z = np.sqrt(np.array(complex(-np.inf, np.nan)))
    with np.errstate(invalid='ignore'):
        if not (np.isnan(z.real) and np.isinf(z.imag)):
            raise AssertionError(msgform % (z.real, z.imag))