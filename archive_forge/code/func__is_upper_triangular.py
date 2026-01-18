import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
def _is_upper_triangular(A):
    if issparse(A):
        lower_part = scipy.sparse.tril(A, -1)
        return lower_part.nnz == 0 or lower_part.count_nonzero() == 0
    elif is_pydata_spmatrix(A):
        import sparse
        lower_part = sparse.tril(A, -1)
        return lower_part.nnz == 0
    else:
        return not np.tril(A, -1).any()