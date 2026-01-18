import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
from .base import OdeSolver, DenseOutput
def _validate_jac(self, jac, sparsity):
    t0 = self.t
    y0 = self.y
    if jac is None:
        if sparsity is not None:
            if issparse(sparsity):
                sparsity = csc_matrix(sparsity)
            groups = group_columns(sparsity)
            sparsity = (sparsity, groups)

        def jac_wrapped(t, y):
            self.njev += 1
            f = self.fun_single(t, y)
            J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f, self.atol, self.jac_factor, sparsity)
            return J
        J = jac_wrapped(t0, y0)
    elif callable(jac):
        J = jac(t0, y0)
        self.njev += 1
        if issparse(J):
            J = csc_matrix(J, dtype=y0.dtype)

            def jac_wrapped(t, y):
                self.njev += 1
                return csc_matrix(jac(t, y), dtype=y0.dtype)
        else:
            J = np.asarray(J, dtype=y0.dtype)

            def jac_wrapped(t, y):
                self.njev += 1
                return np.asarray(jac(t, y), dtype=y0.dtype)
        if J.shape != (self.n, self.n):
            raise ValueError('`jac` is expected to have shape {}, but actually has {}.'.format((self.n, self.n), J.shape))
    else:
        if issparse(jac):
            J = csc_matrix(jac, dtype=y0.dtype)
        else:
            J = np.asarray(jac, dtype=y0.dtype)
        if J.shape != (self.n, self.n):
            raise ValueError('`jac` is expected to have shape {}, but actually has {}.'.format((self.n, self.n), J.shape))
        jac_wrapped = None
    return (jac_wrapped, J)