import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
class TestLeastSq:

    def setup_method(self):
        x = np.linspace(0, 10, 40)
        a, b, c = (3.1, 42, -304.2)
        self.x = x
        self.abc = (a, b, c)
        y_true = a * x ** 2 + b * x + c
        np.random.seed(0)
        self.y_meas = y_true + 0.01 * np.random.standard_normal(y_true.shape)

    def residuals(self, p, y, x):
        a, b, c = p
        err = y - (a * x ** 2 + b * x + c)
        return err

    def residuals_jacobian(self, _p, _y, x):
        return -np.vstack([x ** 2, x, np.ones_like(x)]).T

    def test_basic(self):
        p0 = array([0, 0, 0])
        params_fit, ier = leastsq(self.residuals, p0, args=(self.y_meas, self.x))
        assert_(ier in (1, 2, 3, 4), 'solution not found (ier=%d)' % ier)
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_basic_with_gradient(self):
        p0 = array([0, 0, 0])
        params_fit, ier = leastsq(self.residuals, p0, args=(self.y_meas, self.x), Dfun=self.residuals_jacobian)
        assert_(ier in (1, 2, 3, 4), 'solution not found (ier=%d)' % ier)
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_full_output(self):
        p0 = array([[0, 0, 0]])
        full_output = leastsq(self.residuals, p0, args=(self.y_meas, self.x), full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert_(ier in (1, 2, 3, 4), 'solution not found: %s' % mesg)

    def test_input_untouched(self):
        p0 = array([0, 0, 0], dtype=float64)
        p0_copy = array(p0, copy=True)
        full_output = leastsq(self.residuals, p0, args=(self.y_meas, self.x), full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert_(ier in (1, 2, 3, 4), 'solution not found: %s' % mesg)
        assert_array_equal(p0, p0_copy)

    def test_wrong_shape_func_callable(self):
        func = ReturnShape(1)
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.leastsq, func, x0)

    def test_wrong_shape_func_function(self):
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.leastsq, dummy_func, x0, args=((1,),))

    def test_wrong_shape_Dfun_callable(self):
        func = ReturnShape(1)
        deriv_func = ReturnShape((2, 2))
        assert_raises(TypeError, optimize.leastsq, func, x0=[0, 1], Dfun=deriv_func)

    def test_wrong_shape_Dfun_function(self):

        def func(x):
            return dummy_func(x, (2,))

        def deriv_func(x):
            return dummy_func(x, (3, 3))
        assert_raises(TypeError, optimize.leastsq, func, x0=[0, 1], Dfun=deriv_func)

    def test_float32(self):

        def func(p, x, y):
            q = p[0] * np.exp(-(x - p[1]) ** 2 / (2.0 * p[2] ** 2)) + p[3]
            return q - y
        x = np.array([1.475, 1.429, 1.409, 1.419, 1.455, 1.519, 1.472, 1.368, 1.286, 1.231], dtype=np.float32)
        y = np.array([0.0168, 0.0193, 0.0211, 0.0202, 0.0171, 0.0151, 0.0185, 0.0258, 0.034, 0.0396], dtype=np.float32)
        p0 = np.array([1.0, 1.0, 1.0, 1.0])
        p1, success = optimize.leastsq(func, p0, args=(x, y))
        assert_(success in [1, 2, 3, 4])
        assert_((func(p1, x, y) ** 2).sum() < 0.0001 * (func(p0, x, y) ** 2).sum())

    def test_func_can_raise(self):

        def func(*args):
            raise ValueError('I raised')
        with assert_raises(ValueError, match='I raised'):
            optimize.leastsq(func, x0=[0])

    def test_Dfun_can_raise(self):

        def func(x):
            return x - np.array([10])

        def deriv_func(*args):
            raise ValueError('I raised')
        with assert_raises(ValueError, match='I raised'):
            optimize.leastsq(func, x0=[0], Dfun=deriv_func)

    def test_reentrant_func(self):

        def func(*args):
            self.test_basic()
            return self.residuals(*args)
        p0 = array([0, 0, 0])
        params_fit, ier = leastsq(func, p0, args=(self.y_meas, self.x))
        assert_(ier in (1, 2, 3, 4), 'solution not found (ier=%d)' % ier)
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_reentrant_Dfun(self):

        def deriv_func(*args):
            self.test_basic()
            return self.residuals_jacobian(*args)
        p0 = array([0, 0, 0])
        params_fit, ier = leastsq(self.residuals, p0, args=(self.y_meas, self.x), Dfun=deriv_func)
        assert_(ier in (1, 2, 3, 4), 'solution not found (ier=%d)' % ier)
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_concurrent_no_gradient(self):
        v = sequence_parallel([self.test_basic] * 10)
        assert all([result is None for result in v])

    def test_concurrent_with_gradient(self):
        v = sequence_parallel([self.test_basic_with_gradient] * 10)
        assert all([result is None for result in v])

    def test_func_input_output_length_check(self):

        def func(x):
            return 2 * (x[0] - 3) ** 2 + 1
        with assert_raises(TypeError, match='Improper input: func input vector length N='):
            optimize.leastsq(func, x0=[0, 1])