from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
import operator
def _mul_multivector(self, other):
    result = np.zeros((other.shape[1], self.shape[0]), dtype=upcast_char(self.dtype.char, other.dtype.char))
    for i, col in enumerate(other.T):
        coo_matvec(self.nnz, self.row, self.col, self.data, col, result[i])
    return result.T.view(type=type(other))