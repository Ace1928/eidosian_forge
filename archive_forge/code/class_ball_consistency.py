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
class ball_consistency:
    tol = 0.0

    def distance(self, a, b, p):
        return minkowski_distance(a * 1.0, b * 1.0, p)

    def test_in_ball(self):
        x = np.atleast_2d(self.x)
        d = np.broadcast_to(self.d, x.shape[:-1])
        l = self.T.query_ball_point(x, self.d, p=self.p, eps=self.eps)
        for i, ind in enumerate(l):
            dist = self.distance(self.data[ind], x[i], self.p) - d[i] * (1.0 + self.eps)
            norm = self.distance(self.data[ind], x[i], self.p) + d[i] * (1.0 + self.eps)
            assert_array_equal(dist < self.tol * norm, True)

    def test_found_all(self):
        x = np.atleast_2d(self.x)
        d = np.broadcast_to(self.d, x.shape[:-1])
        l = self.T.query_ball_point(x, self.d, p=self.p, eps=self.eps)
        for i, ind in enumerate(l):
            c = np.ones(self.T.n, dtype=bool)
            c[ind] = False
            dist = self.distance(self.data[c], x[i], self.p) - d[i] / (1.0 + self.eps)
            norm = self.distance(self.data[c], x[i], self.p) + d[i] / (1.0 + self.eps)
            assert_array_equal(dist > -self.tol * norm, True)