import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
class TestPCHIP:

    def _make_random(self, npts=20):
        np.random.seed(1234)
        xi = np.sort(np.random.random(npts))
        yi = np.random.random(npts)
        return (pchip(xi, yi), xi, yi)

    def test_overshoot(self):
        p, xi, yi = self._make_random()
        for i in range(len(xi) - 1):
            x1, x2 = (xi[i], xi[i + 1])
            y1, y2 = (yi[i], yi[i + 1])
            if y1 > y2:
                y1, y2 = (y2, y1)
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert_(((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all())

    def test_monotone(self):
        p, xi, yi = self._make_random()
        for i in range(len(xi) - 1):
            x1, x2 = (xi[i], xi[i + 1])
            y1, y2 = (yi[i], yi[i + 1])
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert_(((y2 - y1) * (yp[1:] - yp[:1]) > 0).all())

    def test_cast(self):
        data = np.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100], [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
        xx = np.arange(100)
        curve = pchip(data[0], data[1])(xx)
        data1 = data * 1.0
        curve1 = pchip(data1[0], data1[1])(xx)
        assert_allclose(curve, curve1, atol=1e-14, rtol=1e-14)

    def test_nag(self):
        dataStr = '\n          7.99   0.00000E+0\n          8.09   0.27643E-4\n          8.19   0.43750E-1\n          8.70   0.16918E+0\n          9.20   0.46943E+0\n         10.00   0.94374E+0\n         12.00   0.99864E+0\n         15.00   0.99992E+0\n         20.00   0.99999E+0\n        '
        data = np.loadtxt(io.StringIO(dataStr))
        pch = pchip(data[:, 0], data[:, 1])
        resultStr = '\n           7.9900       0.0000\n           9.1910       0.4640\n          10.3920       0.9645\n          11.5930       0.9965\n          12.7940       0.9992\n          13.9950       0.9998\n          15.1960       0.9999\n          16.3970       1.0000\n          17.5980       1.0000\n          18.7990       1.0000\n          20.0000       1.0000\n        '
        result = np.loadtxt(io.StringIO(resultStr))
        assert_allclose(result[:, 1], pch(result[:, 0]), rtol=0.0, atol=5e-05)

    def test_endslopes(self):
        x = np.array([0.0, 0.1, 0.25, 0.35])
        y1 = np.array([279.35, 500.0, 1000.0, 2500.0])
        y2 = np.array([279.35, 2500.0, 1500.0, 1000.0])
        for pp in (pchip(x, y1), pchip(x, y2)):
            for t in (x[0], x[-1]):
                assert_(pp(t, 1) != 0)

    def test_all_zeros(self):
        x = np.arange(10)
        y = np.zeros_like(x)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            pch = pchip(x, y)
        xx = np.linspace(0, 9, 101)
        assert_equal(pch(xx), 0.0)

    def test_two_points(self):
        x = np.linspace(0, 1, 11)
        p = pchip([0, 1], [0, 2])
        assert_allclose(p(x), 2 * x, atol=1e-15)

    def test_pchip_interpolate(self):
        assert_array_almost_equal(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=1), [1.0])
        assert_array_almost_equal(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=0), [3.5])
        assert_array_almost_equal(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=[0, 1]), [[3.5], [1]])

    def test_roots(self):
        p = pchip([0, 1], [-1, 1])
        r = p.roots()
        assert_allclose(r, 0.5)