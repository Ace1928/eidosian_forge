import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _unpack_index(index):
    """ Parse index. Always return a tuple of the form (row, col).
    Valid type for row/col is integer, slice, or array of integers.

    Returns:
          resulting row & col indices : single integer, slice, or
          array of integers. If row & column indices are supplied
          explicitly, they are used as the major/minor indices.
          If only one index is supplied, the minor index is
          assumed to be all (e.g., [maj, :]).
    """
    if (isinstance(index, (spmatrix, cupy.ndarray, numpy.ndarray)) or _try_is_scipy_spmatrix(index)) and index.ndim == 2 and (index.dtype.kind == 'b'):
        return index.nonzero()
    index = _eliminate_ellipsis(index)
    if isinstance(index, tuple):
        if len(index) == 2:
            row, col = index
        elif len(index) == 1:
            row, col = (index[0], slice(None))
        else:
            raise IndexError('invalid number of indices')
    else:
        idx = _compatible_boolean_index(index)
        if idx is None:
            row, col = (index, slice(None))
        elif idx.ndim < 2:
            return (_boolean_index_to_array(idx), slice(None))
        elif idx.ndim == 2:
            return idx.nonzero()
    if isspmatrix(row) or isspmatrix(col):
        raise IndexError('Indexing with sparse matrices is not supported except boolean indexing where matrix and index are equal shapes.')
    bool_row = _compatible_boolean_index(row)
    bool_col = _compatible_boolean_index(col)
    if bool_row is not None:
        row = _boolean_index_to_array(bool_row)
    if bool_col is not None:
        col = _boolean_index_to_array(bool_col)
    return (row, col)