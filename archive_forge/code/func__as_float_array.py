import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array.

    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = cupy.asarray(x)
    x = cupy.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and (not cupy.isfinite(x).all()):
        raise ValueError('Array must not contain infs or nans.')
    return x