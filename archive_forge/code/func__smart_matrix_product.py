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
def _smart_matrix_product(A, B, alpha=None, structure=None):
    """
    A matrix product that knows about sparse and structured matrices.

    Parameters
    ----------
    A : 2d ndarray
        First matrix.
    B : 2d ndarray
        Second matrix.
    alpha : float
        The matrix product will be scaled by this constant.
    structure : str, optional
        A string describing the structure of both matrices `A` and `B`.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    M : 2d ndarray
        Matrix product of A and B.

    """
    if len(A.shape) != 2:
        raise ValueError('expected A to be a rectangular matrix')
    if len(B.shape) != 2:
        raise ValueError('expected B to be a rectangular matrix')
    f = None
    if structure == UPPER_TRIANGULAR:
        if not issparse(A) and (not issparse(B)) and (not is_pydata_spmatrix(A)) and (not is_pydata_spmatrix(B)):
            f, = scipy.linalg.get_blas_funcs(('trmm',), (A, B))
    if f is not None:
        if alpha is None:
            alpha = 1.0
        out = f(alpha, A, B)
    elif alpha is None:
        out = A.dot(B)
    else:
        out = alpha * A.dot(B)
    return out