import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _process_deriv_spec(deriv):
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except TypeError as e:
            msg = 'Derivatives, `bc_type`, should be specified as a pair of iterables of pairs of (order, value).'
            raise ValueError(msg) from e
    else:
        ords, vals = ([], [])
    return cupy.atleast_1d(ords, vals)