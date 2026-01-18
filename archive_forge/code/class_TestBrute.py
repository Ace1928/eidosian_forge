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
class TestBrute:

    def setup_method(self):
        self.params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
        self.rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
        self.solution = np.array([-1.05665192, 1.80834843])

    def brute_func(self, z, *params):
        return brute_func(z, *params)

    def test_brute(self):
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params, full_output=True, finish=optimize.fmin)
        assert_allclose(resbrute[0], self.solution, atol=0.001)
        assert_allclose(resbrute[1], brute_func(self.solution, *self.params), atol=0.001)
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params, full_output=True, finish=optimize.minimize)
        assert_allclose(resbrute[0], self.solution, atol=0.001)
        assert_allclose(resbrute[1], brute_func(self.solution, *self.params), atol=0.001)
        resbrute = optimize.brute(self.brute_func, self.rranges, args=self.params, full_output=True, finish=optimize.minimize)
        assert_allclose(resbrute[0], self.solution, atol=0.001)

    def test_1D(self):

        def f(x):
            assert len(x.shape) == 1
            assert x.shape[0] == 1
            return x ** 2
        optimize.brute(f, [(-1, 1)], Ns=3, finish=None)

    def test_workers(self):
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params, full_output=True, finish=None)
        resbrute1 = optimize.brute(brute_func, self.rranges, args=self.params, full_output=True, finish=None, workers=2)
        assert_allclose(resbrute1[-1], resbrute[-1])
        assert_allclose(resbrute1[0], resbrute[0])

    def test_runtime_warning(self):
        rng = np.random.default_rng(1234)

        def func(z, *params):
            return rng.random(1) * 1000
        msg = 'final optimization did not succeed.*|Maximum number of function eval.*'
        with pytest.warns(RuntimeWarning, match=msg):
            optimize.brute(func, self.rranges, args=self.params, disp=True)

    def test_coerce_args_param(self):

        def f(x, *args):
            return x ** args[0]
        resbrute = optimize.brute(f, (slice(-4, 4, 0.25),), args=2)
        assert_allclose(resbrute, 0)