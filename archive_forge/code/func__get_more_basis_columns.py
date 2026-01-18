import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult
def _get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape
    a = np.arange(m + n)
    bl = np.zeros(len(a), dtype=bool)
    bl[basis] = 1
    options = a[~bl]
    options = options[options < n]
    B = np.zeros((m, m))
    B[:, 0:len(basis)] = A[:, basis]
    if basis.size > 0 and np.linalg.matrix_rank(B[:, :len(basis)]) < len(basis):
        raise Exception('Basis has dependent columns')
    rank = 0
    for i in range(n):
        new_basis = np.random.permutation(options)[:m - len(basis)]
        B[:, len(basis):] = A[:, new_basis]
        rank = np.linalg.matrix_rank(B)
        if rank == m:
            break
    return np.concatenate((basis, new_basis))