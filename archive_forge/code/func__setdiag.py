from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
import operator
def _setdiag(self, values, k):
    M, N = self.shape
    if values.ndim and (not len(values)):
        return
    idx_dtype = self.row.dtype
    full_keep = self.col - self.row != k
    if k < 0:
        max_index = min(M + k, N)
        if values.ndim:
            max_index = min(max_index, len(values))
        keep = np.logical_or(full_keep, self.col >= max_index)
        new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
        new_col = np.arange(max_index, dtype=idx_dtype)
    else:
        max_index = min(M, N - k)
        if values.ndim:
            max_index = min(max_index, len(values))
        keep = np.logical_or(full_keep, self.row >= max_index)
        new_row = np.arange(max_index, dtype=idx_dtype)
        new_col = np.arange(k, k + max_index, dtype=idx_dtype)
    if values.ndim:
        new_data = values[:max_index]
    else:
        new_data = np.empty(max_index, dtype=self.dtype)
        new_data[:] = values
    self.row = np.concatenate((self.row[keep], new_row))
    self.col = np.concatenate((self.col[keep], new_col))
    self.data = np.concatenate((self.data[keep], new_data))
    self.has_canonical_format = False