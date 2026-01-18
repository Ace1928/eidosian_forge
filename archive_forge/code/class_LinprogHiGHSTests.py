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
class LinprogHiGHSTests(LinprogCommonTests):

    def test_callback(self):

        def cb(res):
            return None
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]
        assert_raises(NotImplementedError, linprog, c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=self.method)
        _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])

    @pytest.mark.parametrize('options', [{'maxiter': -1}, {'disp': -1}, {'presolve': -1}, {'time_limit': -1}, {'dual_feasibility_tolerance': -1}, {'primal_feasibility_tolerance': -1}, {'ipm_optimality_tolerance': -1}, {'simplex_dual_edge_weight_strategy': 'ekki'}])
    def test_invalid_option_values(self, options):

        def f(options):
            linprog(1, method=self.method, options=options)
        options.update(self.options)
        assert_warns(OptimizeWarning, f, options=options)

    def test_crossover(self):
        A_eq, b_eq, c, _, _ = magic_square(4)
        bounds = (0, 1)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        assert_equal(res.crossover_nit == 0, self.method != 'highs-ipm')

    def test_marginals(self):
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        lb, ub = bounds.T

        def f_bub(x):
            return linprog(c, A_ub, x, A_eq, b_eq, bounds, method=self.method).fun
        dfdbub = approx_derivative(f_bub, b_ub, method='3-point', f0=res.fun)
        assert_allclose(res.ineqlin.marginals, dfdbub)

        def f_beq(x):
            return linprog(c, A_ub, b_ub, A_eq, x, bounds, method=self.method).fun
        dfdbeq = approx_derivative(f_beq, b_eq, method='3-point', f0=res.fun)
        assert_allclose(res.eqlin.marginals, dfdbeq)

        def f_lb(x):
            bounds = np.array([x, ub]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method).fun
        with np.errstate(invalid='ignore'):
            dfdlb = approx_derivative(f_lb, lb, method='3-point', f0=res.fun)
            dfdlb[~np.isfinite(lb)] = 0
        assert_allclose(res.lower.marginals, dfdlb)

        def f_ub(x):
            bounds = np.array([lb, x]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method).fun
        with np.errstate(invalid='ignore'):
            dfdub = approx_derivative(f_ub, ub, method='3-point', f0=res.fun)
            dfdub[~np.isfinite(ub)] = 0
        assert_allclose(res.upper.marginals, dfdub)

    def test_dual_feasibility(self):
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=42)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        resid = -c + A_ub.T @ res.ineqlin.marginals + A_eq.T @ res.eqlin.marginals + res.upper.marginals + res.lower.marginals
        assert_allclose(resid, 0, atol=1e-12)

    def test_complementary_slackness(self):
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=42)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
        assert np.allclose(res.ineqlin.marginals @ (b_ub - A_ub @ res.x), 0)