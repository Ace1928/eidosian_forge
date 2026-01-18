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
class TestSolveSylvester:
    cases = [(np.array([[1, 2], [0, 4]]), np.array([[5, 6], [0, 8]]), np.array([[9, 10], [11, 12]])), (np.array([[1.0, 0, 0, 0], [0, 1.0, 2.0, 0.0], [0, 0, 3.0, -4], [0, 0, 2, 5]]), np.array([[2.0, 0, 0, 1.0], [0, 1.0, 0.0, 0.0], [0, 0, 1.0, -1], [0, 0, 1, 1]]), np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])), (np.array([[1.0 + 1j, 2.0], [3.0 - 4j, 5.0]]), np.array([[-1.0, 2j], [3.0, 4.0]]), np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])), (np.array([[1.0, 2.0], [3.0, 5.0]]), np.array([[-1.0, 0], [3.0, 4.0]]), np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])), (np.array([[1.0 + 1j, 2.0], [3.0 - 4j, 5.0]]), np.array([[-1.0, 0], [3.0, 4.0]]), np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])), (np.array([[1.0 + 1j, 2.0], [3.0 - 4j, 5.0]]), np.array([[-1.0, 0], [3.0, 4.0]]), np.array([[2.0, 2.0], [-1.0, 2.0]])), (np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]), np.array([[2, 3], [4, 5]]), np.array([[1, 2], [3, 4], [5, 6]])), (np.array([[8, 1j, 6 + 2j], [3, 5, 7], [4, 9, 2]]), np.array([[2, 3], [4, 5 - 1j]]), np.array([[1, 2j], [3, 4j], [5j, 6 + 7j]]))]

    def check_case(self, a, b, c):
        x = solve_sylvester(a, b, c)
        assert_array_almost_equal(np.dot(a, x) + np.dot(x, b), c)

    def test_cases(self):
        for case in self.cases:
            self.check_case(case[0], case[1], case[2])

    def test_trivial(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0]])
        c = np.array([2.0, 2.0]).reshape(-1, 1)
        x = solve_sylvester(a, b, c)
        assert_array_almost_equal(x, np.array([1.0, 1.0]).reshape(-1, 1))