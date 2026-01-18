import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
class TestLinearNDInterpolation:

    def test_smoketest(self):
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        yi = interpnd.LinearNDInterpolator(x, y)(x)
        assert_almost_equal(y, yi)

    def test_smoketest_alternate(self):
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        yi = interpnd.LinearNDInterpolator((x[:, 0], x[:, 1]), y)(x[:, 0], x[:, 1])
        assert_almost_equal(y, yi)

    def test_complex_smoketest(self):
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        yi = interpnd.LinearNDInterpolator(x, y)(x)
        assert_almost_equal(y, yi)

    def test_tri_input(self):
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        yi = interpnd.LinearNDInterpolator(tri, y)(x)
        assert_almost_equal(y, yi)

    def test_square(self):
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
        values = np.array([1.0, 2.0, -3.0, 5.0], dtype=np.float64)

        def ip(x, y):
            t1 = x + y <= 1
            t2 = ~t1
            x1 = x[t1]
            y1 = y[t1]
            x2 = x[t2]
            y2 = y[t2]
            z = 0 * x
            z[t1] = values[0] * (1 - x1 - y1) + values[1] * y1 + values[3] * x1
            z[t2] = values[2] * (x2 + y2 - 1) + values[1] * (1 - x2) + values[3] * (1 - y2)
            return z
        xx, yy = np.broadcast_arrays(np.linspace(0, 1, 14)[:, None], np.linspace(0, 1, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = np.array([xx, yy]).T.copy()
        zi = interpnd.LinearNDInterpolator(points, values)(xi)
        assert_almost_equal(zi, ip(xx, yy))

    def test_smoketest_rescale(self):
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        yi = interpnd.LinearNDInterpolator(x, y, rescale=True)(x)
        assert_almost_equal(y, yi)

    def test_square_rescale(self):
        points = np.array([(0, 0), (0, 100), (10, 100), (10, 0)], dtype=np.float64)
        values = np.array([1.0, 2.0, -3.0, 5.0], dtype=np.float64)
        xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:, None], np.linspace(0, 100, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = np.array([xx, yy]).T.copy()
        zi = interpnd.LinearNDInterpolator(points, values)(xi)
        zi_rescaled = interpnd.LinearNDInterpolator(points, values, rescale=True)(xi)
        assert_almost_equal(zi, zi_rescaled)

    def test_tripoints_input_rescale(self):
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        yi = interpnd.LinearNDInterpolator(tri.points, y)(x)
        yi_rescale = interpnd.LinearNDInterpolator(tri.points, y, rescale=True)(x)
        assert_almost_equal(yi, yi_rescale)

    def test_tri_input_rescale(self):
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        match = 'Rescaling is not supported when passing a Delaunay triangulation as ``points``.'
        with pytest.raises(ValueError, match=match):
            interpnd.LinearNDInterpolator(tri, y, rescale=True)(x)

    def test_pickle(self):
        np.random.seed(1234)
        x = np.random.rand(30, 2)
        y = np.random.rand(30) + 1j * np.random.rand(30)
        ip = interpnd.LinearNDInterpolator(x, y)
        ip2 = pickle.loads(pickle.dumps(ip))
        assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))