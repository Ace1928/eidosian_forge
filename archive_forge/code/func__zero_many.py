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
def _zero_many(self, i, j):
    """Sets value at each (i, j) to zero, preserving sparsity structure.
        Here (i,j) index major and minor respectively.
        """
    i, j, M, N = self._prepare_indices(i, j)
    new_sp = cupyx.scipy.sparse.csr_matrix((cupy.arange(self.nnz, dtype=cupy.float32), self.indices, self.indptr), shape=(M, N))
    offsets = new_sp._get_arrayXarray(i, j, not_found_val=-1).astype(cupy.int32).ravel()
    self.data[offsets[offsets > -1]] = 0