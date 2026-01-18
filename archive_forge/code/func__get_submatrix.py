from warnings import warn
import operator
import numpy as np
from scipy._lib._util import _prune_array
from ._base import _spbase, issparse, SparseEfficiencyWarning
from ._data import _data_matrix, _minmax_mixin
from . import _sparsetools
from ._sparsetools import (get_csr_submatrix, csr_sample_offsets, csr_todense,
from ._index import IndexMixin
from ._sputils import (upcast, upcast_char, to_native, isdense, isshape,
def _get_submatrix(self, major=None, minor=None, copy=False):
    """Return a submatrix of this matrix.

        major, minor: None, int, or slice with step 1
        """
    M, N = self._swap(self.shape)
    i0, i1 = _process_slice(major, M)
    j0, j1 = _process_slice(minor, N)
    if i0 == 0 and j0 == 0 and (i1 == M) and (j1 == N):
        return self.copy() if copy else self
    indptr, indices, data = get_csr_submatrix(M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)
    shape = self._swap((i1 - i0, j1 - j0))
    return self.__class__((data, indices, indptr), shape=shape, dtype=self.dtype, copy=False)