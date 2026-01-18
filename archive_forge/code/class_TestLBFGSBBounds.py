import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
class TestLBFGSBBounds:

    def setup_method(self):
        self.bounds = ((1, None), (None, None))
        self.solution = (1, 0)

    def fun(self, x, p=2.0):
        return 1.0 / p * (x[0] ** p + x[1] ** p)

    def jac(self, x, p=2.0):
        return x ** (p - 1)

    def fj(self, x, p=2.0):
        return (self.fun(x, p), self.jac(x, p))

    def test_l_bfgs_b_bounds(self):
        x, f, d = optimize.fmin_l_bfgs_b(self.fun, [0, -1], fprime=self.jac, bounds=self.bounds)
        assert d['warnflag'] == 0, d['task']
        assert_allclose(x, self.solution, atol=1e-06)

    def test_l_bfgs_b_funjac(self):
        x, f, d = optimize.fmin_l_bfgs_b(self.fj, [0, -1], args=(2.0,), bounds=self.bounds)
        assert d['warnflag'] == 0, d['task']
        assert_allclose(x, self.solution, atol=1e-06)

    def test_minimize_l_bfgs_b_bounds(self):
        res = optimize.minimize(self.fun, [0, -1], method='L-BFGS-B', jac=self.jac, bounds=self.bounds)
        assert res['success'], res['message']
        assert_allclose(res.x, self.solution, atol=1e-06)

    @pytest.mark.parametrize('bounds', [[(10, 1), (1, 10)], [(1, 10), (10, 1)], [(10, 1), (10, 1)]])
    def test_minimize_l_bfgs_b_incorrect_bounds(self, bounds):
        with pytest.raises(ValueError, match='.*bound.*'):
            optimize.minimize(self.fun, [0, -1], method='L-BFGS-B', jac=self.jac, bounds=bounds)

    def test_minimize_l_bfgs_b_bounds_FD(self):
        jacs = ['2-point', '3-point', None]
        argss = [(2.0,), ()]
        for jac, args in itertools.product(jacs, argss):
            res = optimize.minimize(self.fun, [0, -1], args=args, method='L-BFGS-B', jac=jac, bounds=self.bounds, options={'finite_diff_rel_step': None})
            assert res['success'], res['message']
            assert_allclose(res.x, self.solution, atol=1e-06)