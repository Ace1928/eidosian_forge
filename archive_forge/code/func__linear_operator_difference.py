import functools
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator
from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
from ._group_columns import group_dense, group_sparse
from scipy._lib._array_api import atleast_nd, array_namespace
def _linear_operator_difference(fun, x0, f0, h, method):
    m = f0.size
    n = x0.size
    if method == '2-point':

        def matvec(p):
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            dx = h / norm(p)
            x = x0 + dx * p
            df = fun(x) - f0
            return df / dx
    elif method == '3-point':

        def matvec(p):
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            dx = 2 * h / norm(p)
            x1 = x0 - dx / 2 * p
            x2 = x0 + dx / 2 * p
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1
            return df / dx
    elif method == 'cs':

        def matvec(p):
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            dx = h / norm(p)
            x = x0 + dx * p * 1j
            f1 = fun(x)
            df = f1.imag
            return df / dx
    else:
        raise RuntimeError('Never be here.')
    return LinearOperator((m, n), matvec)