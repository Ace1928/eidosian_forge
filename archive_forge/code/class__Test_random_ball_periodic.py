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
@KDTreeTest
class _Test_random_ball_periodic(ball_consistency):

    def distance(self, a, b, p):
        return distance_box(a, b, p, 1.0)

    def setup_method(self):
        n = 10000
        m = 4
        np.random.seed(1234)
        self.data = np.random.uniform(size=(n, m))
        self.T = self.kdtree_type(self.data, leafsize=2, boxsize=1)
        self.x = np.full(m, 0.1)
        self.p = 2.0
        self.eps = 0
        self.d = 0.2

    def test_in_ball_outside(self):
        l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
        for i in l:
            assert_(self.distance(self.data[i], self.x, self.p) <= self.d * (1.0 + self.eps))
        l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
        for i in l:
            assert_(self.distance(self.data[i], self.x, self.p) <= self.d * (1.0 + self.eps))

    def test_found_all_outside(self):
        c = np.ones(self.T.n, dtype=bool)
        l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
        c[l] = False
        assert np.all(self.distance(self.data[c], self.x, self.p) >= self.d / (1.0 + self.eps))
        l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
        c[l] = False
        assert np.all(self.distance(self.data[c], self.x, self.p) >= self.d / (1.0 + self.eps))