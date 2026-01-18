from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
class TestQuadraticFunction:

    def setup_method(self):
        self.J = np.array([[0.1, 0.2], [-1.0, 1.0], [0.5, 0.2]])
        self.g = np.array([0.8, -2.0])
        self.diag = np.array([1.0, 2.0])

    def test_build_quadratic_1d(self):
        s = np.zeros(2)
        a, b = build_quadratic_1d(self.J, self.g, s)
        assert_equal(a, 0)
        assert_equal(b, 0)
        a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
        assert_equal(a, 0)
        assert_equal(b, 0)
        s = np.array([1.0, -1.0])
        a, b = build_quadratic_1d(self.J, self.g, s)
        assert_equal(a, 2.05)
        assert_equal(b, 2.8)
        a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
        assert_equal(a, 3.55)
        assert_equal(b, 2.8)
        s0 = np.array([0.5, 0.5])
        a, b, c = build_quadratic_1d(self.J, self.g, s, diag=self.diag, s0=s0)
        assert_equal(a, 3.55)
        assert_allclose(b, 2.39)
        assert_allclose(c, -0.1525)

    def test_minimize_quadratic_1d(self):
        a = 5
        b = -1
        t, y = minimize_quadratic_1d(a, b, 1, 2)
        assert_equal(t, 1)
        assert_allclose(y, a * t ** 2 + b * t, rtol=1e-15)
        t, y = minimize_quadratic_1d(a, b, -2, -1)
        assert_equal(t, -1)
        assert_allclose(y, a * t ** 2 + b * t, rtol=1e-15)
        t, y = minimize_quadratic_1d(a, b, -1, 1)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t ** 2 + b * t, rtol=1e-15)
        c = 10
        t, y = minimize_quadratic_1d(a, b, -1, 1, c=c)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
        t, y = minimize_quadratic_1d(a, b, -np.inf, np.inf, c=c)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
        t, y = minimize_quadratic_1d(a, b, 0, np.inf, c=c)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
        t, y = minimize_quadratic_1d(a, b, -np.inf, 0, c=c)
        assert_equal(t, 0)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
        a = -1
        b = 0.2
        t, y = minimize_quadratic_1d(a, b, -np.inf, np.inf)
        assert_equal(y, -np.inf)
        t, y = minimize_quadratic_1d(a, b, 0, np.inf)
        assert_equal(t, np.inf)
        assert_equal(y, -np.inf)
        t, y = minimize_quadratic_1d(a, b, -np.inf, 0)
        assert_equal(t, -np.inf)
        assert_equal(y, -np.inf)

    def test_evaluate_quadratic(self):
        s = np.array([1.0, -1.0])
        value = evaluate_quadratic(self.J, self.g, s)
        assert_equal(value, 4.85)
        value = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
        assert_equal(value, 6.35)
        s = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
        values = evaluate_quadratic(self.J, self.g, s)
        assert_allclose(values, [4.85, -0.91, 0.0])
        values = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
        assert_allclose(values, [6.35, 0.59, 0.0])