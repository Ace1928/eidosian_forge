from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
import operator
def _getnnz(self, axis=None):
    if axis is None:
        nnz = len(self.data)
        if nnz != len(self.row) or nnz != len(self.col):
            raise ValueError('row, column, and data array must all be the same length')
        if self.data.ndim != 1 or self.row.ndim != 1 or self.col.ndim != 1:
            raise ValueError('row, column, and data arrays must be 1-D')
        return int(nnz)
    if axis < 0:
        axis += 2
    if axis == 0:
        return np.bincount(downcast_intp_index(self.col), minlength=self.shape[1])
    elif axis == 1:
        return np.bincount(downcast_intp_index(self.row), minlength=self.shape[0])
    else:
        raise ValueError('axis out of bounds')