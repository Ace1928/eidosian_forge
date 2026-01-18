import string
import warnings
import numpy
import cupy
import cupyx
from cupy import _core
from cupy._core import _scalar
from cupy._creation import basic
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _index
def _perform_insert(self, indices_inserts, data_inserts, rows, row_counts, idx_dtype):
    """Insert new elements into current sparse matrix in sorted order"""
    indptr_diff = cupy.diff(self.indptr)
    indptr_diff[rows] += row_counts
    new_indptr = cupy.empty(self.indptr.shape, dtype=idx_dtype)
    new_indptr[0] = idx_dtype(0)
    new_indptr[1:] = indptr_diff
    cupy.cumsum(new_indptr, out=new_indptr)
    out_nnz = int(new_indptr[-1])
    new_indices = cupy.empty(out_nnz, dtype=idx_dtype)
    new_data = cupy.empty(out_nnz, dtype=self.data.dtype)
    new_indptr_lookup = cupy.zeros(new_indptr.size, dtype=idx_dtype)
    new_indptr_lookup[1:][rows] = row_counts
    cupy.cumsum(new_indptr_lookup, out=new_indptr_lookup)
    _index._insert_many_populate_arrays(indices_inserts, data_inserts, new_indptr_lookup, self.indptr, self.indices, self.data, new_indptr, new_indices, new_data, size=self.indptr.size - 1)
    self.indptr = new_indptr
    self.indices = new_indices
    self.data = new_data