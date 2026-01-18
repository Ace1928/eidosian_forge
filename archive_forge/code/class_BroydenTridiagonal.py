from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
class BroydenTridiagonal:

    def __init__(self, n=100, mode='sparse'):
        np.random.seed(0)
        self.n = n
        self.x0 = -np.ones(n)
        self.lb = np.linspace(-2, -1.5, n)
        self.ub = np.linspace(-0.8, 0.0, n)
        self.lb += 0.1 * np.random.randn(n)
        self.ub += 0.1 * np.random.randn(n)
        self.x0 += 0.1 * np.random.randn(n)
        self.x0 = make_strictly_feasible(self.x0, self.lb, self.ub)
        if mode == 'sparse':
            self.sparsity = lil_matrix((n, n), dtype=int)
            i = np.arange(n)
            self.sparsity[i, i] = 1
            i = np.arange(1, n)
            self.sparsity[i, i - 1] = 1
            i = np.arange(n - 1)
            self.sparsity[i, i + 1] = 1
            self.jac = self._jac
        elif mode == 'operator':
            self.jac = lambda x: aslinearoperator(self._jac(x))
        elif mode == 'dense':
            self.sparsity = None
            self.jac = lambda x: self._jac(x).toarray()
        else:
            assert_(False)

    def fun(self, x):
        f = (3 - x) * x + 1
        f[1:] -= x[:-1]
        f[:-1] -= 2 * x[1:]
        return f

    def _jac(self, x):
        J = lil_matrix((self.n, self.n))
        i = np.arange(self.n)
        J[i, i] = 3 - 2 * x
        i = np.arange(1, self.n)
        J[i, i - 1] = -1
        i = np.arange(self.n - 1)
        J[i, i + 1] = -2
        return J