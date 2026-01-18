import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = c.size
    zs = np.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1:] = c / 2
    return zs + zs[::-1]