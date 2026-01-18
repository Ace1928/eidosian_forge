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
def _divide_sparse(self, other):
    """
        Divide this matrix by a second sparse matrix.
        """
    if other.shape != self.shape:
        raise ValueError('inconsistent shapes')
    r = self._binopt(other, '_eldiv_')
    if np.issubdtype(r.dtype, np.inexact):
        out = np.empty(self.shape, dtype=self.dtype)
        out.fill(np.nan)
        row, col = other.nonzero()
        out[row, col] = 0
        r = r.tocoo()
        out[r.row, r.col] = r.data
        out = self._container(out)
    else:
        out = r
    return out