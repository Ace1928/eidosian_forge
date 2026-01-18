import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _periodic_knots(x, k):
    """Returns vector of nodes on a circle."""
    xc = cupy.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = cupy.diff(xc)
        xc[1:-1] -= dx[:-1] / 2
    dx = cupy.diff(xc)
    t = cupy.zeros(n + 2 * k)
    t[k:-k] = xc
    for i in range(0, k):
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t