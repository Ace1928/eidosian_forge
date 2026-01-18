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
def _solve_P_Q(U, V, structure=None):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade numerator.
    V : ndarray
        Pade denominator.
    structure : str, optional
        A string describing the structure of both matrices `U` and `V`.
        Only `upper_triangular` is currently supported.

    Notes
    -----
    The `structure` argument is inspired by similar args
    for theano and cvxopt functions.

    """
    P = U + V
    Q = -U + V
    if issparse(U) or is_pydata_spmatrix(U):
        return spsolve(Q, P)
    elif structure is None:
        return solve(Q, P)
    elif structure == UPPER_TRIANGULAR:
        return solve_triangular(Q, P)
    else:
        raise ValueError('unsupported matrix structure: ' + str(structure))