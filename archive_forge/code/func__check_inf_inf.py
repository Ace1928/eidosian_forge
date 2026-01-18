import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def _check_inf_inf(dummy):
    msgform = 'cexp(inf, inf) is (%f, %f), expected (+-inf, nan)'
    with np.errstate(invalid='ignore'):
        z = f(np.array(complex(np.inf, np.inf)))
        if not np.isinf(z.real) or not np.isnan(z.imag):
            raise AssertionError(msgform % (z.real, z.imag))