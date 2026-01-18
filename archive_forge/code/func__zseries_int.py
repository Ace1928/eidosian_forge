import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def _zseries_int(zs):
    """Integrate a z-series.

    The integral is with respect to x, not z. This is achieved by a change
    of variable using dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to integrate

    Returns
    -------
    integral : z-series
        The indefinite integral

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    dividing the resulting zs by two.

    """
    n = 1 + len(zs) // 2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    div = np.arange(-n, n + 1) * 2
    zs[:n] /= div[:n]
    zs[n + 1:] /= div[n + 1:]
    zs[n] = 0
    return zs