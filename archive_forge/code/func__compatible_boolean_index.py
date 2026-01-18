import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _compatible_boolean_index(idx):
    """Returns a boolean index array that can be converted to
    integer array. Returns None if no such array exists.
    """
    if hasattr(idx, 'ndim'):
        if idx.dtype.kind == 'b':
            return idx
    elif _first_element_bool(idx):
        return cupy.asarray(idx, dtype='bool')
    return None