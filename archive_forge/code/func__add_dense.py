from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
import operator
def _add_dense(self, other):
    if other.shape != self.shape:
        raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
    dtype = upcast_char(self.dtype.char, other.dtype.char)
    result = np.array(other, dtype=dtype, copy=True)
    fortran = int(result.flags.f_contiguous)
    M, N = self.shape
    coo_todense(M, N, self.nnz, self.row, self.col, self.data, result.ravel('A'), fortran)
    return self._container(result, copy=False)