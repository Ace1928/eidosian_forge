import math
import cupy
from cupy.linalg import lstsq
from cupyx.scipy.ndimage import convolve1d
from ._arraytools import axis_slice
def _polyder(p, m):
    """Differentiate polynomials represented with coefficients.

    p must be a 1-D or 2-D array.  In the 2-D case, each column gives
    the coefficients of a polynomial; the first row holds the coefficients
    associated with the highest power. m must be a nonnegative integer.
    (numpy.polyder doesn't handle the 2-D case.)
    """
    if m == 0:
        result = p
    else:
        n = len(p)
        if n <= m:
            result = cupy.zeros_like(p[:1, ...])
        else:
            dp = p[:-m].copy()
            for k in range(m):
                rng = cupy.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result