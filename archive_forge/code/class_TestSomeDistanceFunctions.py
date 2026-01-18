import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
class TestSomeDistanceFunctions:

    def setup_method(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])
        self.cases = [(x, y)]

    def test_minkowski(self):
        for x, y in self.cases:
            dist1 = minkowski(x, y, p=1)
            assert_almost_equal(dist1, 3.0)
            dist1p5 = minkowski(x, y, p=1.5)
            assert_almost_equal(dist1p5, (1.0 + 2.0 ** 1.5) ** (2.0 / 3))
            dist2 = minkowski(x, y, p=2)
            assert_almost_equal(dist2, 5.0 ** 0.5)
            dist0p25 = minkowski(x, y, p=0.25)
            assert_almost_equal(dist0p25, (1.0 + 2.0 ** 0.25) ** 4)
        a = np.array([352, 916])
        b = np.array([350, 660])
        assert_equal(minkowski(a, b), minkowski(a.astype('uint16'), b.astype('uint16')))

    def test_euclidean(self):
        for x, y in self.cases:
            dist = weuclidean(x, y)
            assert_almost_equal(dist, np.sqrt(5))

    def test_sqeuclidean(self):
        for x, y in self.cases:
            dist = wsqeuclidean(x, y)
            assert_almost_equal(dist, 5.0)

    def test_cosine(self):
        for x, y in self.cases:
            dist = wcosine(x, y)
            assert_almost_equal(dist, 1.0 - 18.0 / (np.sqrt(14) * np.sqrt(27)))

    def test_correlation(self):
        xm = np.array([-1.0, 0, 1.0])
        ym = np.array([-4.0 / 3, -4.0 / 3, 5.0 - 7.0 / 3])
        for x, y in self.cases:
            dist = wcorrelation(x, y)
            assert_almost_equal(dist, 1.0 - np.dot(xm, ym) / (norm(xm) * norm(ym)))

    def test_correlation_positive(self):
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, -2.0, 0.0, -2.0, 0.0, 0.0, -1.0, -2.0, 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0])
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 2.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])
        dist = correlation(x, y)
        assert 0 <= dist <= 10 * np.finfo(np.float64).eps

    def test_mahalanobis(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])
        vi = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        for x, y in self.cases:
            dist = mahalanobis(x, y, vi)
            assert_almost_equal(dist, np.sqrt(6.0))