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
def _fragment_2_1(X, T, s):
    """
    A helper function for expm_2009.

    Notes
    -----
    The argument X is modified in-place, but this modification is not the same
    as the returned value of the function.
    This function also takes pains to do things in ways that are compatible
    with sparse matrices, for example by avoiding fancy indexing
    and by using methods of the matrices whenever possible instead of
    using functions of the numpy or scipy libraries themselves.

    """
    n = X.shape[0]
    diag_T = np.ravel(T.diagonal().copy())
    scale = 2 ** (-s)
    exp_diag = np.exp(scale * diag_T)
    for k in range(n):
        X[k, k] = exp_diag[k]
    for i in range(s - 1, -1, -1):
        X = X.dot(X)
        scale = 2 ** (-i)
        exp_diag = np.exp(scale * diag_T)
        for k in range(n):
            X[k, k] = exp_diag[k]
        for k in range(n - 1):
            lam_1 = scale * diag_T[k]
            lam_2 = scale * diag_T[k + 1]
            t_12 = scale * T[k, k + 1]
            value = _eq_10_42(lam_1, lam_2, t_12)
            X[k, k + 1] = value
    return X