import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
class TestCubicSpline:

    @staticmethod
    def check_correctness(S, bc_start='not-a-knot', bc_end='not-a-knot', tol=1e-14):
        """Check that spline coefficients satisfy the continuity and boundary
        conditions."""
        x = S.x
        c = S.c
        dx = np.diff(x)
        dx = dx.reshape([dx.shape[0]] + [1] * (c.ndim - 2))
        dxi = dx[:-1]
        assert_allclose(c[3, 1:], c[0, :-1] * dxi ** 3 + c[1, :-1] * dxi ** 2 + c[2, :-1] * dxi + c[3, :-1], rtol=tol, atol=tol)
        assert_allclose(c[2, 1:], 3 * c[0, :-1] * dxi ** 2 + 2 * c[1, :-1] * dxi + c[2, :-1], rtol=tol, atol=tol)
        assert_allclose(c[1, 1:], 3 * c[0, :-1] * dxi + c[1, :-1], rtol=tol, atol=tol)
        if x.size == 3 and bc_start == 'not-a-knot' and (bc_end == 'not-a-knot'):
            assert_allclose(c[0], 0, rtol=tol, atol=tol)
            return
        if bc_start == 'periodic':
            assert_allclose(S(x[0], 0), S(x[-1], 0), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 1), S(x[-1], 1), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 2), S(x[-1], 2), rtol=tol, atol=tol)
            return
        if bc_start == 'not-a-knot':
            if x.size == 2:
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[0], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, 0], c[0, 1], rtol=tol, atol=tol)
        elif bc_start == 'clamped':
            assert_allclose(S(x[0], 1), 0, rtol=tol, atol=tol)
        elif bc_start == 'natural':
            assert_allclose(S(x[0], 2), 0, rtol=tol, atol=tol)
        else:
            order, value = bc_start
            assert_allclose(S(x[0], order), value, rtol=tol, atol=tol)
        if bc_end == 'not-a-knot':
            if x.size == 2:
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[1], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, -1], c[0, -2], rtol=tol, atol=tol)
        elif bc_end == 'clamped':
            assert_allclose(S(x[-1], 1), 0, rtol=tol, atol=tol)
        elif bc_end == 'natural':
            assert_allclose(S(x[-1], 2), 0, rtol=2 * tol, atol=2 * tol)
        else:
            order, value = bc_end
            assert_allclose(S(x[-1], order), value, rtol=tol, atol=tol)

    def check_all_bc(self, x, y, axis):
        deriv_shape = list(y.shape)
        del deriv_shape[axis]
        first_deriv = np.empty(deriv_shape)
        first_deriv.fill(2)
        second_deriv = np.empty(deriv_shape)
        second_deriv.fill(-1)
        bc_all = ['not-a-knot', 'natural', 'clamped', (1, first_deriv), (2, second_deriv)]
        for bc in bc_all[:3]:
            S = CubicSpline(x, y, axis=axis, bc_type=bc)
            self.check_correctness(S, bc, bc)
        for bc_start in bc_all:
            for bc_end in bc_all:
                S = CubicSpline(x, y, axis=axis, bc_type=(bc_start, bc_end))
                self.check_correctness(S, bc_start, bc_end, tol=2e-14)

    def test_general(self):
        x = np.array([-1, 0, 0.5, 2, 4, 4.5, 5.5, 9])
        y = np.array([0, -0.5, 2, 3, 2.5, 1, 1, 0.5])
        for n in [2, 3, x.size]:
            self.check_all_bc(x[:n], y[:n], 0)
            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y[:n]
            Y[0, :, 1] = y[:n] - 1
            Y[1, :, 0] = y[:n] + 2
            Y[1, :, 1] = y[:n] + 3
            self.check_all_bc(x[:n], Y, 1)

    def test_periodic(self):
        for n in [2, 3, 5]:
            x = np.linspace(0, 2 * np.pi, n)
            y = np.cos(x)
            S = CubicSpline(x, y, bc_type='periodic')
            self.check_correctness(S, 'periodic', 'periodic')
            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y
            Y[0, :, 1] = y + 2
            Y[1, :, 0] = y - 1
            Y[1, :, 1] = y + 5
            S = CubicSpline(x, Y, axis=1, bc_type='periodic')
            self.check_correctness(S, 'periodic', 'periodic')

    def test_periodic_eval(self):
        x = np.linspace(0, 2 * np.pi, 10)
        y = np.cos(x)
        S = CubicSpline(x, y, bc_type='periodic')
        assert_almost_equal(S(1), S(1 + 2 * np.pi), decimal=15)

    def test_second_derivative_continuity_gh_11758(self):
        x = np.array([0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0, 7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3])
        y = np.array([1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1, 2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 1.3])
        S = CubicSpline(x, y, bc_type='periodic', extrapolate='periodic')
        self.check_correctness(S, 'periodic', 'periodic')

    def test_three_points(self):
        x = np.array([1.0, 2.75, 3.0])
        y = np.array([1.0, 15.0, 1.0])
        S = CubicSpline(x, y, bc_type='periodic')
        self.check_correctness(S, 'periodic', 'periodic')
        assert_allclose(S.derivative(1)(x), np.array([-48.0, -48.0, -48.0]))

    def test_dtypes(self):
        x = np.array([0, 1, 2, 3], dtype=int)
        y = np.array([-5, 2, 3, 1], dtype=int)
        S = CubicSpline(x, y)
        self.check_correctness(S)
        y = np.array([-1 + 1j, 0.0, 1 - 1j, 0.5 - 1.5j])
        S = CubicSpline(x, y)
        self.check_correctness(S)
        S = CubicSpline(x, x ** 3, bc_type=('natural', (1, 2j)))
        self.check_correctness(S, 'natural', (1, 2j))
        y = np.array([-5, 2, 3, 1])
        S = CubicSpline(x, y, bc_type=[(1, 2 + 0.5j), (2, 0.5 - 1j)])
        self.check_correctness(S, (1, 2 + 0.5j), (2, 0.5 - 1j))

    def test_small_dx(self):
        rng = np.random.RandomState(0)
        x = np.sort(rng.uniform(size=100))
        y = 10000.0 + rng.uniform(size=100)
        S = CubicSpline(x, y)
        self.check_correctness(S, tol=1e-13)

    def test_incorrect_inputs(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        xc = np.array([1 + 1j, 2, 3, 4])
        xn = np.array([np.nan, 2, 3, 4])
        xo = np.array([2, 1, 3, 4])
        yn = np.array([np.nan, 2, 3, 4])
        y3 = [1, 2, 3]
        x1 = [1]
        y1 = [1]
        assert_raises(ValueError, CubicSpline, xc, y)
        assert_raises(ValueError, CubicSpline, xn, y)
        assert_raises(ValueError, CubicSpline, x, yn)
        assert_raises(ValueError, CubicSpline, xo, y)
        assert_raises(ValueError, CubicSpline, x, y3)
        assert_raises(ValueError, CubicSpline, x[:, np.newaxis], y)
        assert_raises(ValueError, CubicSpline, x1, y1)
        wrong_bc = [('periodic', 'clamped'), ((2, 0), (3, 10)), ((1, 0),), (0.0, 0.0), 'not-a-typo']
        for bc_type in wrong_bc:
            assert_raises(ValueError, CubicSpline, x, y, 0, bc_type, True)
        Y = np.c_[y, y]
        bc1 = ('clamped', (1, 0))
        bc2 = ('clamped', (1, [0, 0, 0]))
        bc3 = ('clamped', (1, [[0, 0]]))
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc1, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc2, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc3, True)
        assert_raises(ValueError, CubicSpline, x, y, 0, 'periodic', True)