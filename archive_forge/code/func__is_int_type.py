import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def _is_int_type(x):
    """
    Check if input is of a scalar integer type (so ``5`` and ``array(5)`` will
    pass, while ``5.0`` and ``array([5])`` will fail.
    """
    if cupy.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except TypeError:
        return False
    else:
        return True