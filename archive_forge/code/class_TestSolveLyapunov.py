import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.linalg import solve_sylvester
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.linalg import block_diag, solve, LinAlgError
from scipy.sparse._sputils import matrix
class TestSolveLyapunov:
    cases = [(np.array([[1, 2], [3, 4]]), np.array([[9, 10], [11, 12]])), (np.array([[1.0 + 1j, 2.0], [3.0 - 4j, 5.0]]), np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])), (np.array([[1.0, 2.0], [3.0, 5.0]]), np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])), (np.array([[1.0 + 1j, 2.0], [3.0 - 4j, 5.0]]), np.array([[2.0, 2.0], [-1.0, 2.0]])), (np.array([[3, 9, 5, 1, 4], [1, 2, 3, 8, 4], [4, 6, 6, 6, 3], [1, 5, 2, 0, 7], [5, 3, 3, 1, 5]]), np.array([[2, 4, 1, 0, 1], [4, 1, 0, 2, 0], [1, 0, 3, 0, 3], [0, 2, 0, 1, 0], [1, 0, 3, 0, 4]])), (np.array([[0.1 + 0j, 0.091 + 0j, 0.082 + 0j, 0.073 + 0j, 0.064 + 0j, 0.055 + 0j, 0.046 + 0j, 0.037 + 0j, 0.028 + 0j, 0.019 + 0j, 0.01 + 0j], [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j]]), np.eye(11)), (matrix([[0, 1], [-1 / 2, -1]]), matrix([0, 3]).T @ matrix([0, 3]).T.T), (matrix([[0, 1], [-1 / 2, -1]]), np.array(matrix([0, 3]).T @ matrix([0, 3]).T.T))]

    def test_continuous_squareness_and_shape(self):
        nsq = np.ones((3, 2))
        sq = np.eye(3)
        assert_raises(ValueError, solve_continuous_lyapunov, nsq, sq)
        assert_raises(ValueError, solve_continuous_lyapunov, sq, nsq)
        assert_raises(ValueError, solve_continuous_lyapunov, sq, np.eye(2))

    def check_continuous_case(self, a, q):
        x = solve_continuous_lyapunov(a, q)
        assert_array_almost_equal(np.dot(a, x) + np.dot(x, a.conj().transpose()), q)

    def check_discrete_case(self, a, q, method=None):
        x = solve_discrete_lyapunov(a, q, method=method)
        assert_array_almost_equal(np.dot(np.dot(a, x), a.conj().transpose()) - x, -1.0 * q)

    def test_cases(self):
        for case in self.cases:
            self.check_continuous_case(case[0], case[1])
            self.check_discrete_case(case[0], case[1])
            self.check_discrete_case(case[0], case[1], method='direct')
            self.check_discrete_case(case[0], case[1], method='bilinear')