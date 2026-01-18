import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def _check_ninf_inf(dummy):
    msgform = 'cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)'
    with np.errstate(invalid='ignore'):
        z = f(np.array(complex(-np.inf, np.inf)))
        if z.real != 0 or z.imag != 0:
            raise AssertionError(msgform % (z.real, z.imag))