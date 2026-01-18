import numpy as np
from ._matrix import spmatrix
from ._base import issparse, _formats, _spbase, sparray
from ._data import _data_matrix
from ._sputils import (
from ._sparsetools import dia_matvec
def _data_mask(self):
    """Returns a mask of the same shape as self.data, where
        mask[i,j] is True when data[i,j] corresponds to a stored element."""
    num_rows, num_cols = self.shape
    offset_inds = np.arange(self.data.shape[1])
    row = offset_inds - self.offsets[:, None]
    mask = row >= 0
    mask &= row < num_rows
    mask &= offset_inds < num_cols
    return mask