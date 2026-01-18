import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
def _integrate(c, x, a, b, extrapolate, out):
    """
    Compute integral over a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    a : double
        Start point of integration.
    b : double
        End point of integration.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (n,)
        Integral of the piecewise polynomial, assuming the polynomial
        is zero outside the range (x[0], x[-1]).
        This argument is modified in-place.
    """
    ascending = x[-1] >= x[0]
    a = cupy.asarray([a], dtype=cupy.float64)
    b = cupy.asarray([b], dtype=cupy.float64)
    start_interval = cupy.empty(a.shape, dtype=cupy.int64)
    end_interval = cupy.empty(b.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position')
    interval_kernel(((a.shape[0] + 128 - 1) // 128,), (128,), (x, a, start_interval, extrapolate, a.shape[0], x.shape[0], ascending))
    interval_kernel(((b.shape[0] + 128 - 1) // 128,), (128,), (x, b, end_interval, extrapolate, b.shape[0], x.shape[0], ascending))
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    int_kernel = _get_module_func(PPOLY_MODULE, 'integrate', c)
    int_kernel(((c.shape[2] + 128 - 1) // 128,), (128,), (c, x, a, b, start_interval, end_interval, c_shape, c_strides, ascending, out))