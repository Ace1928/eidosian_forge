from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _getrow(self, i):
    """Returns a copy of row i of the array, as a (1 x n) sparse
        array (row vector).
        """
    m = self.shape[0]
    if i < 0:
        i += m
    if i < 0 or i >= m:
        raise IndexError('index out of bounds')
    row_selector = self._csr_container(([1], [[0], [i]]), shape=(1, m), dtype=self.dtype)
    return row_selector @ self