import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _not_a_knot(x, k):
    """Given data x, construct the knot vector w/ not-a-knot BC.
    cf de Boor, XIII(12)."""
    x = cupy.asarray(x)
    if k % 2 != 1:
        raise ValueError('Odd degree for now only. Got %s.' % k)
    m = (k - 1) // 2
    t = x[m + 1:-m - 1]
    t = cupy.r_[(x[0],) * (k + 1), t, (x[-1],) * (k + 1)]
    return t