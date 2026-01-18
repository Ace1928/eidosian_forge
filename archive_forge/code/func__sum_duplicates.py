from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
import operator
def _sum_duplicates(self, row, col, data):
    if len(data) == 0:
        return (row, col, data)
    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    data = data[order]
    unique_mask = (row[1:] != row[:-1]) | (col[1:] != col[:-1])
    unique_mask = np.append(True, unique_mask)
    row = row[unique_mask]
    col = col[unique_mask]
    unique_inds, = np.nonzero(unique_mask)
    data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
    return (row, col, data)