import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def _normed_hermite_e_n(x, n):
    """
    Evaluate a normalized HermiteE polynomial.

    Compute the value of the normalized HermiteE polynomial of degree ``n``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    n : int
        Degree of the normalized HermiteE function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    .. versionadded:: 1.10.0

    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard HermiteE functions
    overflow when n >= 207.

    """
    if n == 0:
        return np.full(x.shape, 1 / np.sqrt(np.sqrt(2 * np.pi)))
    c0 = 0.0
    c1 = 1.0 / np.sqrt(np.sqrt(2 * np.pi))
    nd = float(n)
    for i in range(n - 1):
        tmp = c0
        c0 = -c1 * np.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * np.sqrt(1.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x