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
class TestFixedPoint:

    def test_scalar_trivial(self):

        def func(x):
            return 2.0 * x
        x0 = 1.0
        x = fixed_point(func, x0)
        assert_almost_equal(x, 0.0)

    def test_scalar_basic1(self):

        def func(x):
            return x ** 2
        x0 = 1.05
        x = fixed_point(func, x0)
        assert_almost_equal(x, 1.0)

    def test_scalar_basic2(self):

        def func(x):
            return x ** 0.5
        x0 = 1.05
        x = fixed_point(func, x0)
        assert_almost_equal(x, 1.0)

    def test_array_trivial(self):

        def func(x):
            return 2.0 * x
        x0 = [0.3, 0.15]
        with np.errstate(all='ignore'):
            x = fixed_point(func, x0)
        assert_almost_equal(x, [0.0, 0.0])

    def test_array_basic1(self):

        def func(x, c):
            return c * x ** 2
        c = array([0.75, 1.0, 1.25])
        x0 = [1.1, 1.15, 0.9]
        with np.errstate(all='ignore'):
            x = fixed_point(func, x0, args=(c,))
        assert_almost_equal(x, 1.0 / c)

    def test_array_basic2(self):

        def func(x, c):
            return c * x ** 0.5
        c = array([0.75, 1.0, 1.25])
        x0 = [0.8, 1.1, 1.1]
        x = fixed_point(func, x0, args=(c,))
        assert_almost_equal(x, c ** 2)

    def test_lambertw(self):
        xxroot = fixed_point(lambda xx: np.exp(-2.0 * xx) / 2.0, 1.0, args=(), xtol=1e-12, maxiter=500)
        assert_allclose(xxroot, np.exp(-2.0 * xxroot) / 2.0)
        assert_allclose(xxroot, lambertw(1) / 2)

    def test_no_acceleration(self):
        ks = 2
        kl = 6
        m = 1.3
        n0 = 1.001
        i0 = (m - 1) / m * (kl / ks / m) ** (1 / (m - 1))

        def func(n):
            return np.log(kl / ks / n) / np.log(i0 * n / (n - 1)) + 1
        n = fixed_point(func, n0, method='iteration')
        assert_allclose(n, m)