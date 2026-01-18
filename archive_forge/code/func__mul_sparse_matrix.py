from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _mul_sparse_matrix(self, other):
    return self.tocsr()._mul_sparse_matrix(other)