import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
class TestApproxDerivativesDense:

    def fun_scalar_scalar(self, x):
        return np.sinh(x)

    def jac_scalar_scalar(self, x):
        return np.cosh(x)

    def fun_scalar_vector(self, x):
        return np.array([x[0] ** 2, np.tan(x[0]), np.exp(x[0])])

    def jac_scalar_vector(self, x):
        return np.array([2 * x[0], np.cos(x[0]) ** (-2), np.exp(x[0])]).reshape(-1, 1)

    def fun_vector_scalar(self, x):
        return np.sin(x[0] * x[1]) * np.log(x[0])

    def wrong_dimensions_fun(self, x):
        return np.array([x ** 2, np.tan(x), np.exp(x)])

    def jac_vector_scalar(self, x):
        return np.array([x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) + np.sin(x[0] * x[1]) / x[0], x[0] * np.cos(x[0] * x[1]) * np.log(x[0])])

    def fun_vector_vector(self, x):
        return np.array([x[0] * np.sin(x[1]), x[1] * np.cos(x[0]), x[0] ** 3 * x[1] ** (-0.5)])

    def jac_vector_vector(self, x):
        return np.array([[np.sin(x[1]), x[0] * np.cos(x[1])], [-x[1] * np.sin(x[0]), np.cos(x[0])], [3 * x[0] ** 2 * x[1] ** (-0.5), -0.5 * x[0] ** 3 * x[1] ** (-1.5)]])

    def fun_parametrized(self, x, c0, c1=1.0):
        return np.array([np.exp(c0 * x[0]), np.exp(c1 * x[1])])

    def jac_parametrized(self, x, c0, c1=0.1):
        return np.array([[c0 * np.exp(c0 * x[0]), 0], [0, c1 * np.exp(c1 * x[1])]])

    def fun_with_nan(self, x):
        return x if np.abs(x) <= 1e-08 else np.nan

    def jac_with_nan(self, x):
        return 1.0 if np.abs(x) <= 1e-08 else np.nan

    def fun_zero_jacobian(self, x):
        return np.array([x[0] * x[1], np.cos(x[0] * x[1])])

    def jac_zero_jacobian(self, x):
        return np.array([[x[1], x[0]], [-x[1] * np.sin(x[0] * x[1]), -x[0] * np.sin(x[0] * x[1])]])

    def jac_non_numpy(self, x):
        xp = np.asarray(x).item()
        return math.exp(xp)

    def test_scalar_scalar(self):
        x0 = 1.0
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0, method='2-point')
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0)
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0, method='cs')
        jac_true = self.jac_scalar_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-09)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_scalar_scalar_abs_step(self):
        x0 = 1.0
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0, method='2-point', abs_step=1.49e-08)
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0, abs_step=1.49e-08)
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0, method='cs', abs_step=1.49e-08)
        jac_true = self.jac_scalar_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-09)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_scalar_vector(self):
        x0 = 0.5
        jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0, method='2-point')
        jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0)
        jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0, method='cs')
        jac_true = self.jac_scalar_vector(np.atleast_1d(x0))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-09)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_scalar(self):
        x0 = np.array([100.0, -0.5])
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0, method='2-point')
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0)
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0, method='cs')
        jac_true = self.jac_vector_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-07)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_scalar_abs_step(self):
        x0 = np.array([100.0, -0.5])
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0, method='2-point', abs_step=1.49e-08)
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0, abs_step=1.49e-08, rel_step=np.inf)
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0, method='cs', abs_step=1.49e-08)
        jac_true = self.jac_vector_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=3e-09)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_vector(self):
        x0 = np.array([-100.0, 0.2])
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0, method='2-point')
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0)
        jac_diff_4 = approx_derivative(self.fun_vector_vector, x0, method='cs')
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-05)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_wrong_dimensions(self):
        x0 = 1.0
        assert_raises(RuntimeError, approx_derivative, self.wrong_dimensions_fun, x0)
        f0 = self.wrong_dimensions_fun(np.atleast_1d(x0))
        assert_raises(ValueError, approx_derivative, self.wrong_dimensions_fun, x0, f0=f0)

    def test_custom_rel_step(self):
        x0 = np.array([-0.1, 0.1])
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0, method='2-point', rel_step=0.0001)
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0, rel_step=0.0001)
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=0.01)
        assert_allclose(jac_diff_3, jac_true, rtol=0.0001)

    def test_options(self):
        x0 = np.array([1.0, 1.0])
        c0 = -1.0
        c1 = 1.0
        lb = 0.0
        ub = 2.0
        f0 = self.fun_parametrized(x0, c0, c1=c1)
        rel_step = np.array([-1e-06, 1e-07])
        jac_true = self.jac_parametrized(x0, c0, c1)
        jac_diff_2 = approx_derivative(self.fun_parametrized, x0, method='2-point', rel_step=rel_step, f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
        jac_diff_3 = approx_derivative(self.fun_parametrized, x0, rel_step=rel_step, f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-09)

    def test_with_bounds_2_point(self):
        lb = -np.ones(2)
        ub = np.ones(2)
        x0 = np.array([-2.0, 0.2])
        assert_raises(ValueError, approx_derivative, self.fun_vector_vector, x0, bounds=(lb, ub))
        x0 = np.array([-1.0, 1.0])
        jac_diff = approx_derivative(self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-06)

    def test_with_bounds_3_point(self):
        lb = np.array([1.0, 1.0])
        ub = np.array([2.0, 2.0])
        x0 = np.array([1.0, 2.0])
        jac_true = self.jac_vector_vector(x0)
        jac_diff = approx_derivative(self.fun_vector_vector, x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-09)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(lb, np.inf))
        assert_allclose(jac_diff, jac_true, rtol=1e-09)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(-np.inf, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-09)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-09)

    def test_tight_bounds(self):
        x0 = np.array([10.0, 10.0])
        lb = x0 - 3e-09
        ub = x0 + 2e-09
        jac_true = self.jac_vector_vector(x0)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-06)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, method='2-point', rel_step=1e-06, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-06)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-06)
        jac_diff = approx_derivative(self.fun_vector_vector, x0, rel_step=1e-06, bounds=(lb, ub))
        assert_allclose(jac_true, jac_diff, rtol=1e-06)

    def test_bound_switches(self):
        lb = -1e-08
        ub = 1e-08
        x0 = 0.0
        jac_true = self.jac_with_nan(x0)
        jac_diff_2 = approx_derivative(self.fun_with_nan, x0, method='2-point', rel_step=1e-06, bounds=(lb, ub))
        jac_diff_3 = approx_derivative(self.fun_with_nan, x0, rel_step=1e-06, bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-09)
        x0 = 1e-08
        jac_true = self.jac_with_nan(x0)
        jac_diff_2 = approx_derivative(self.fun_with_nan, x0, method='2-point', rel_step=1e-06, bounds=(lb, ub))
        jac_diff_3 = approx_derivative(self.fun_with_nan, x0, rel_step=1e-06, bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-09)

    def test_non_numpy(self):
        x0 = 1.0
        jac_true = self.jac_non_numpy(x0)
        jac_diff_2 = approx_derivative(self.jac_non_numpy, x0, method='2-point')
        jac_diff_3 = approx_derivative(self.jac_non_numpy, x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-08)
        assert_raises(TypeError, approx_derivative, self.jac_non_numpy, x0, **dict(method='cs'))

    def test_fp(self):
        np.random.seed(1)

        def func(p, x):
            return p[0] + p[1] * x

        def err(p, x, y):
            return func(p, x) - y
        x = np.linspace(0, 1, 100, dtype=np.float64)
        y = np.random.random(100).astype(np.float64)
        p0 = np.array([-1.0, -1.0])
        jac_fp64 = approx_derivative(err, p0, method='2-point', args=(x, y))
        jac_fp = approx_derivative(err, p0.astype(np.float32), method='2-point', args=(x, y))
        assert err(p0, x, y).dtype == np.float64
        assert_allclose(jac_fp, jac_fp64, atol=0.001)

        def err_fp32(p):
            assert p.dtype == np.float32
            return err(p, x, y).astype(np.float32)
        jac_fp = approx_derivative(err_fp32, p0.astype(np.float32), method='2-point')
        assert_allclose(jac_fp, jac_fp64, atol=0.001)

        def f(x):
            return np.sin(x)

        def g(x):
            return np.cos(x)

        def hess(x):
            return -np.sin(x)

        def calc_atol(h, x0, f, hess, EPS):
            t0 = h / 2 * max(np.abs(hess(x0)), np.abs(hess(x0 + h)))
            t1 = EPS / h * max(np.abs(f(x0)), np.abs(f(x0 + h)))
            return t0 + t1
        for dtype in [np.float16, np.float32, np.float64]:
            EPS = np.finfo(dtype).eps
            x0 = np.array(1.0).astype(dtype)
            h = _compute_absolute_step(None, x0, f(x0), '2-point')
            atol = calc_atol(h, x0, f, hess, EPS)
            err = approx_derivative(f, x0, method='2-point', abs_step=h) - g(x0)
            assert abs(err) < atol

    def test_check_derivative(self):
        x0 = np.array([-10.0, 10])
        accuracy = check_derivative(self.fun_vector_vector, self.jac_vector_vector, x0)
        assert_(accuracy < 1e-09)
        accuracy = check_derivative(self.fun_vector_vector, self.jac_vector_vector, x0)
        assert_(accuracy < 1e-06)
        x0 = np.array([0.0, 0.0])
        accuracy = check_derivative(self.fun_zero_jacobian, self.jac_zero_jacobian, x0)
        assert_(accuracy == 0)
        accuracy = check_derivative(self.fun_zero_jacobian, self.jac_zero_jacobian, x0)
        assert_(accuracy == 0)