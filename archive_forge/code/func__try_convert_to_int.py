import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def _try_convert_to_int(x):
    """Return an integer for ``5`` and ``array(5)``, fail if not an
       integer scalar.

    NB: would be easier if ``operator.index(cupy.array(5))`` worked
    (numpy.array(5) does)
    """
    if isinstance(x, cupy.ndarray):
        if x.ndim == 0:
            value = x.item()
        else:
            return (x, False)
    else:
        value = x
    try:
        return (operator.index(value), True)
    except TypeError:
        return (value, False)