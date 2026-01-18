import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def _raise_typeerror(X):
    """Raises a TypeError if X is not a CSR or CSC matrix"""
    input_type = X.format if sp.issparse(X) else type(X)
    err = 'Expected a CSR or CSC sparse matrix, got %s.' % input_type
    raise TypeError(err)