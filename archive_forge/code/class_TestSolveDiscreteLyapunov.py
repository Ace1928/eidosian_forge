import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestSolveDiscreteLyapunov:

    def solve_dicrete_lyapunov_direct(self, a, q, complex_step=False):
        if not complex_step:
            lhs = np.kron(a, a.conj())
            lhs = np.eye(lhs.shape[0]) - lhs
            x = np.linalg.solve(lhs, q.flatten())
        else:
            lhs = np.kron(a, a)
            lhs = np.eye(lhs.shape[0]) - lhs
            x = np.linalg.solve(lhs, q.flatten())
        return np.reshape(x, q.shape)

    def test_univariate(self):
        a = np.array([[0.5]])
        q = np.array([[10.0]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)
        a = np.array([[0.5 + 1j]])
        q = np.array([[10.0]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)
        a = np.array([[0.5 + 1j]])
        q = np.array([[10.0]])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)

    def test_multivariate(self):
        a = tools.companion_matrix([1, -0.4, 0.5])
        q = np.diag([10.0, 5.0])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)
        a = tools.companion_matrix([1, -0.4 + 0.1j, 0.5])
        q = np.diag([10.0, 5.0])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=False)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=False)
        assert_allclose(actual, desired)
        a = tools.companion_matrix([1, -0.4 + 0.1j, 0.5])
        q = np.diag([10.0, 5.0])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)