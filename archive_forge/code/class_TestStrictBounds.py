import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
class TestStrictBounds(TestCase):

    def test_scalarvalue_unique_enforce_feasibility(self):
        m = 3
        lb = 2
        ub = 4
        enforce_feasibility = False
        strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
        assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])
        assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])
        enforce_feasibility = True
        strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
        assert_array_equal(strict_lb, [2, 2, 2])
        assert_array_equal(strict_ub, [4, 4, 4])

    def test_vectorvalue_unique_enforce_feasibility(self):
        m = 3
        lb = [1, 2, 3]
        ub = [4, 5, 6]
        enforce_feasibility = False
        strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
        assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])
        assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])
        enforce_feasibility = True
        strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
        assert_array_equal(strict_lb, [1, 2, 3])
        assert_array_equal(strict_ub, [4, 5, 6])

    def test_scalarvalue_vector_enforce_feasibility(self):
        m = 3
        lb = 2
        ub = 4
        enforce_feasibility = [False, True, False]
        strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
        assert_array_equal(strict_lb, [-np.inf, 2, -np.inf])
        assert_array_equal(strict_ub, [np.inf, 4, np.inf])

    def test_vectorvalue_vector_enforce_feasibility(self):
        m = 3
        lb = [1, 2, 3]
        ub = [4, 6, np.inf]
        enforce_feasibility = [True, False, True]
        strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
        assert_array_equal(strict_lb, [1, -np.inf, 3])
        assert_array_equal(strict_ub, [4, np.inf, np.inf])