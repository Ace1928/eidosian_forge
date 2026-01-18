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
def _set_many(self, i, j, x):
    """Sets value at each (i, j) to x
        Here (i,j) index major and minor respectively, and must not contain
        duplicate entries.
        """
    i, j, M, N = self._prepare_indices(i, j)
    x = cupy.array(x, dtype=self.dtype, copy=True, ndmin=1).ravel()
    new_sp = cupyx.scipy.sparse.csr_matrix((cupy.arange(self.nnz, dtype=cupy.float32), self.indices, self.indptr), shape=(M, N))
    offsets = new_sp._get_arrayXarray(i, j, not_found_val=-1).astype(cupy.int32).ravel()
    if -1 not in offsets:
        self.data[offsets] = x
        return
    else:
        warnings.warn('Changing the sparsity structure of a {}_matrix is expensive.'.format(self.format), _base.SparseEfficiencyWarning)
        mask = offsets > -1
        self.data[offsets[mask]] = x[mask]
        mask = ~mask
        i = i[mask]
        i[i < 0] += M
        j = j[mask]
        j[j < 0] += N
        self._insert_many(i, j, x[mask])