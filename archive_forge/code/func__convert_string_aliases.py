import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _convert_string_aliases(deriv, target_shape):
    if isinstance(deriv, str):
        if deriv == 'clamped':
            deriv = [(1, cupy.zeros(target_shape))]
        elif deriv == 'natural':
            deriv = [(2, cupy.zeros(target_shape))]
        else:
            raise ValueError('Unknown boundary condition : %s' % deriv)
    return deriv