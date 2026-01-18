from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _getcol(self, j):
    """Returns a copy of column j of the array, as an (m x 1) sparse
        array (column vector).
        """
    n = self.shape[1]
    if j < 0:
        j += n
    if j < 0 or j >= n:
        raise IndexError('index out of bounds')
    col_selector = self._csc_container(([1], [[j], [0]]), shape=(n, 1), dtype=self.dtype)
    return self @ col_selector