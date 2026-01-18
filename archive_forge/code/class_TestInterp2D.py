from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
class TestInterp2D:

    def test_interp2d(self):
        y, x = mgrid[0:2:20j, 0:pi:21j]
        z = sin(x + 0.5 * y)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            II = interp2d(x, y, z)
            assert_almost_equal(II(1.0, 2.0), sin(2.0), decimal=2)
            v, u = ogrid[0:2:24j, 0:pi:25j]
            assert_almost_equal(II(u.ravel(), v.ravel()), sin(u + 0.5 * v), decimal=2)

    def test_interp2d_meshgrid_input(self):
        x = linspace(0, 2, 16)
        y = linspace(0, pi, 21)
        z = sin(x[None, :] + y[:, None] / 2.0)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            II = interp2d(x, y, z)
            assert_almost_equal(II(1.0, 2.0), sin(2.0), decimal=2)

    def test_interp2d_meshgrid_input_unsorted(self):
        np.random.seed(1234)
        x = linspace(0, 2, 16)
        y = linspace(0, pi, 21)
        z = sin(x[None, :] + y[:, None] / 2.0)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            ip1 = interp2d(x.copy(), y.copy(), z, kind='cubic')
            np.random.shuffle(x)
            z = sin(x[None, :] + y[:, None] / 2.0)
            ip2 = interp2d(x.copy(), y.copy(), z, kind='cubic')
            np.random.shuffle(x)
            np.random.shuffle(y)
            z = sin(x[None, :] + y[:, None] / 2.0)
            ip3 = interp2d(x, y, z, kind='cubic')
            x = linspace(0, 2, 31)
            y = linspace(0, pi, 30)
            assert_equal(ip1(x, y), ip2(x, y))
            assert_equal(ip1(x, y), ip3(x, y))

    def test_interp2d_eval_unsorted(self):
        y, x = mgrid[0:2:20j, 0:pi:21j]
        z = sin(x + 0.5 * y)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            func = interp2d(x, y, z)
            xe = np.array([3, 4, 5])
            ye = np.array([5.3, 7.1])
            assert_allclose(func(xe, ye), func(xe, ye[::-1]))
            assert_raises(ValueError, func, xe, ye[::-1], 0, 0, True)

    def test_interp2d_linear(self):
        a = np.zeros([5, 5])
        a[2, 2] = 1.0
        x = y = np.arange(5)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            b = interp2d(x, y, a, 'linear')
            assert_almost_equal(b(2.0, 1.5), np.array([0.5]), decimal=2)
            assert_almost_equal(b(2.0, 2.5), np.array([0.5]), decimal=2)

    def test_interp2d_bounds(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 2, 7)
        z = x[None, :] ** 2 + y[:, None]
        ix = np.linspace(-1, 3, 31)
        iy = np.linspace(-1, 3, 33)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            b = interp2d(x, y, z, bounds_error=True)
            assert_raises(ValueError, b, ix, iy)
            b = interp2d(x, y, z, fill_value=np.nan)
            iz = b(ix, iy)
            mx = (ix < 0) | (ix > 1)
            my = (iy < 0) | (iy > 2)
            assert_(np.isnan(iz[my, :]).all())
            assert_(np.isnan(iz[:, mx]).all())
            assert_(np.isfinite(iz[~my, :][:, ~mx]).all())