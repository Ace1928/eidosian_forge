import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
@staticmethod
def _raise_degree(c, d):
    """
        Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1-D
        d : integer

        Returns
        -------
        array
            coefficient array, 1-D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \\sum_{j=0}^{d} b_{a+j, k+d} \\
                                 comb(d, j) / comb(k+d, a+j)
        """
    if d == 0:
        return c
    k = c.shape[0] - 1
    out = cupy.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)
    for a in range(c.shape[0]):
        f = c[a] * _comb(k, a)
        for j in range(d + 1):
            out[a + j] += f * _comb(d, j) / _comb(k + d, a + j)
    return out