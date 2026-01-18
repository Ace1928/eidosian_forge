import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
class TestAutoscaleRS(AutoscaleTests):
    method = 'revised simplex'

    def test_nontrivial_problem_with_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bad_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bad_guess = [1, 2, 3, 0.5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=bad_guess)
        assert_equal(res.status, 6)