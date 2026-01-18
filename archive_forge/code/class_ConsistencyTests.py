import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy.spatial._ckdtree import cKDTreeNode
from scipy.spatial import minkowski_distance
import itertools
class ConsistencyTests:

    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    def test_nearest(self):
        x = self.x
        d, i = self.kdtree.query(x, 1)
        assert_almost_equal(d ** 2, np.sum((x - self.data[i]) ** 2))
        eps = 1e-08
        assert_(np.all(np.sum((self.data - x[np.newaxis, :]) ** 2, axis=1) > d ** 2 - eps))

    def test_m_nearest(self):
        x = self.x
        m = self.m
        dd, ii = self.kdtree.query(x, m)
        d = np.amax(dd)
        i = ii[np.argmax(dd)]
        assert_almost_equal(d ** 2, np.sum((x - self.data[i]) ** 2))
        eps = 1e-08
        assert_equal(np.sum(np.sum((self.data - x[np.newaxis, :]) ** 2, axis=1) < d ** 2 + eps), m)

    def test_points_near(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, distance_upper_bound=d)
        eps = 1e-08
        hits = 0
        for near_d, near_i in zip(dd, ii):
            if near_d == np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d ** 2, np.sum((x - self.data[near_i]) ** 2))
            assert_(near_d < d + eps, f'near_d={near_d:g} should be less than {d:g}')
        assert_equal(np.sum(self.distance(self.data, x, 2) < d ** 2 + eps), hits)

    def test_points_near_l1(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=1, distance_upper_bound=d)
        eps = 1e-08
        hits = 0
        for near_d, near_i in zip(dd, ii):
            if near_d == np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i], 1))
            assert_(near_d < d + eps, f'near_d={near_d:g} should be less than {d:g}')
        assert_equal(np.sum(self.distance(self.data, x, 1) < d + eps), hits)

    def test_points_near_linf(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=np.inf, distance_upper_bound=d)
        eps = 1e-08
        hits = 0
        for near_d, near_i in zip(dd, ii):
            if near_d == np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i], np.inf))
            assert_(near_d < d + eps, f'near_d={near_d:g} should be less than {d:g}')
        assert_equal(np.sum(self.distance(self.data, x, np.inf) < d + eps), hits)

    def test_approx(self):
        x = self.x
        k = self.k
        eps = 0.1
        d_real, i_real = self.kdtree.query(x, k)
        d, i = self.kdtree.query(x, k, eps=eps)
        assert_(np.all(d <= d_real * (1 + eps)))