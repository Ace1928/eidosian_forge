import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _select_last_indices(i, j, x, idx_dtype):
    """Find the unique indices for each row and keep only the last"""
    i = cupy.asarray(i, dtype=idx_dtype)
    j = cupy.asarray(j, dtype=idx_dtype)
    stacked = cupy.stack([j, i])
    order = cupy.lexsort(stacked).astype(idx_dtype)
    indptr_inserts = i[order]
    indices_inserts = j[order]
    data_inserts = x[order]
    mask = cupy.ones(indptr_inserts.size, dtype='bool')
    _unique_mask_kern(indptr_inserts, indices_inserts, order, mask, size=indptr_inserts.size - 1)
    return (indptr_inserts[mask], indices_inserts[mask], data_inserts[mask])