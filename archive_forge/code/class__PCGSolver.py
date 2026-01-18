from functools import partial
import networkx as nx
from networkx.utils import (
class _PCGSolver:
    """Preconditioned conjugate gradient method.

    To solve Ax = b:
        M = A.diagonal() # or some other preconditioner
        solver = _PCGSolver(lambda x: A * x, lambda x: M * x)
        x = solver.solve(b)

    The inputs A and M are functions which compute
    matrix multiplication on the argument.
    A - multiply by the matrix A in Ax=b
    M - multiply by M, the preconditioner surrogate for A

    Warning: There is no limit on number of iterations.
    """

    def __init__(self, A, M):
        self._A = A
        self._M = M

    def solve(self, B, tol):
        import numpy as np
        B = np.asarray(B)
        X = np.ndarray(B.shape, order='F')
        for j in range(B.shape[1]):
            X[:, j] = self._solve(B[:, j], tol)
        return X

    def _solve(self, b, tol):
        import numpy as np
        import scipy as sp
        A = self._A
        M = self._M
        tol *= sp.linalg.blas.dasum(b)
        x = np.zeros(b.shape)
        r = b.copy()
        z = M(r)
        rz = sp.linalg.blas.ddot(r, z)
        p = z.copy()
        while True:
            Ap = A(p)
            alpha = rz / sp.linalg.blas.ddot(p, Ap)
            x = sp.linalg.blas.daxpy(p, x, a=alpha)
            r = sp.linalg.blas.daxpy(Ap, r, a=-alpha)
            if sp.linalg.blas.dasum(r) < tol:
                return x
            z = M(r)
            beta = sp.linalg.blas.ddot(r, z)
            beta, rz = (beta / rz, beta)
            p = sp.linalg.blas.daxpy(p, z, a=beta)