import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
class TestLinearConstraint:

    def test_defaults(self):
        A = np.eye(4)
        lc = LinearConstraint(A)
        lc2 = LinearConstraint(A, -np.inf, np.inf)
        assert_array_equal(lc.lb, lc2.lb)
        assert_array_equal(lc.ub, lc2.ub)

    def test_input_validation(self):
        A = np.eye(4)
        message = '`lb`, `ub`, and `keep_feasible` must be broadcastable'
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A, [1, 2], [1, 2, 3])
        message = 'Constraint limits must be dense arrays'
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A, sps.coo_array([1, 2]), [2, 3])
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A, [1, 2], sps.coo_array([2, 3]))
        message = '`keep_feasible` must be a dense array'
        with pytest.raises(ValueError, match=message):
            keep_feasible = sps.coo_array([True, True])
            LinearConstraint(A, [1, 2], [2, 3], keep_feasible=keep_feasible)
        A = np.empty((4, 3, 5))
        message = '`A` must have exactly two dimensions.'
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A)

    def test_residual(self):
        A = np.eye(2)
        lc = LinearConstraint(A, -2, 4)
        x0 = [-1, 2]
        np.testing.assert_allclose(lc.residual(x0), ([1, 4], [5, 2]))