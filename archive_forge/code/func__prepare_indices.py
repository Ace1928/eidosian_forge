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
def _prepare_indices(self, i, j):
    M, N = self._swap(*self.shape)

    def check_bounds(indices, bound):
        idx = indices.max()
        if idx >= bound:
            raise IndexError('index (%d) out of range (>= %d)' % (idx, bound))
        idx = indices.min()
        if idx < -bound:
            raise IndexError('index (%d) out of range (< -%d)' % (idx, bound))
    i = cupy.array(i, dtype=self.indptr.dtype, copy=True, ndmin=1).ravel()
    j = cupy.array(j, dtype=self.indices.dtype, copy=True, ndmin=1).ravel()
    check_bounds(i, M)
    check_bounds(j, N)
    return (i, j, M, N)