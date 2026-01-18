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
def _major_index_fancy(self, idx):
    """Index along the major axis where idx is an array of ints.
        """
    _, N = self._swap(*self.shape)
    M = idx.size
    new_shape = self._swap(M, N)
    if self.nnz == 0 or M == 0:
        return self.__class__(new_shape, dtype=self.dtype)
    return self.__class__(_index._csr_row_index(self.data, self.indices, self.indptr, idx), shape=new_shape, copy=False)