import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
def _fix_continuity(c, x, order):
    """
    Make a piecewise polynomial continuously differentiable to given order.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.

        Coefficients c[-order-1:] are modified in-place.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    order : int
        Order up to which enforce piecewise differentiability.
    """
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    continuity_kernel = _get_module_func(PPOLY_MODULE, 'fix_continuity', c)
    continuity_kernel((1,), (1,), (c, x, order, c_shape, c_strides, x.shape[0]))