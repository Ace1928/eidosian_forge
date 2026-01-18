import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
class Test_HalfspaceIntersection:

    def assert_unordered_allclose(self, arr1, arr2, rtol=1e-07):
        """Check that every line in arr1 is only once in arr2"""
        assert_equal(arr1.shape, arr2.shape)
        truths = np.zeros((arr1.shape[0],), dtype=bool)
        for l1 in arr1:
            indexes = np.nonzero((abs(arr2 - l1) < rtol).all(axis=1))[0]
            assert_equal(indexes.shape, (1,))
            truths[indexes[0]] = True
        assert_(truths.all())

    @pytest.mark.parametrize('dt', [np.float64, int])
    def test_cube_halfspace_intersection(self, dt):
        halfspaces = np.array([[-1, 0, 0], [0, -1, 0], [1, 0, -2], [0, 1, -2]], dtype=dt)
        feasible_point = np.array([1, 1], dtype=dt)
        points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        hull = qhull.HalfspaceIntersection(halfspaces, feasible_point)
        assert_allclose(hull.intersections, points)

    def test_self_dual_polytope_intersection(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'selfdual-4d-polytope.txt')
        ineqs = np.genfromtxt(fname)
        halfspaces = -np.hstack((ineqs[:, 1:], ineqs[:, :1]))
        feas_point = np.array([0.0, 0.0, 0.0, 0.0])
        hs = qhull.HalfspaceIntersection(halfspaces, feas_point)
        assert_equal(hs.intersections.shape, (24, 4))
        assert_almost_equal(hs.dual_volume, 32.0)
        assert_equal(len(hs.dual_facets), 24)
        for facet in hs.dual_facets:
            assert_equal(len(facet), 6)
        dists = halfspaces[:, -1] + halfspaces[:, :-1].dot(feas_point)
        self.assert_unordered_allclose((halfspaces[:, :-1].T / dists).T, hs.dual_points)
        points = itertools.permutations([0.0, 0.0, 0.5, -0.5])
        for point in points:
            assert_equal(np.sum((hs.intersections == point).all(axis=1)), 1)

    def test_wrong_feasible_point(self):
        halfspaces = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
        feasible_point = np.array([0.5, 0.5, 0.5])
        assert_raises(ValueError, qhull.HalfspaceIntersection, halfspaces, feasible_point)
        feasible_point = np.array([[0.5], [0.5]])
        assert_raises(ValueError, qhull.HalfspaceIntersection, halfspaces, feasible_point)
        feasible_point = np.array([[0.5, 0.5]])
        assert_raises(ValueError, qhull.HalfspaceIntersection, halfspaces, feasible_point)
        feasible_point = np.array([-0.5, -0.5])
        assert_raises(qhull.QhullError, qhull.HalfspaceIntersection, halfspaces, feasible_point)

    def test_incremental(self):
        halfspaces = np.array([[0.0, 0.0, -1.0, -0.5], [0.0, -1.0, 0.0, -0.5], [-1.0, 0.0, 0.0, -0.5], [1.0, 0.0, 0.0, -0.5], [0.0, 1.0, 0.0, -0.5], [0.0, 0.0, 1.0, -0.5]])
        extra_normals = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1, -1.0, -1.0]])
        offsets = np.array([[-1.0]] * 8)
        extra_halfspaces = np.hstack((np.vstack((extra_normals, -extra_normals)), offsets))
        feas_point = np.array([0.0, 0.0, 0.0])
        inc_hs = qhull.HalfspaceIntersection(halfspaces, feas_point, incremental=True)
        inc_res_hs = qhull.HalfspaceIntersection(halfspaces, feas_point, incremental=True)
        for i, ehs in enumerate(extra_halfspaces):
            inc_hs.add_halfspaces(ehs[np.newaxis, :])
            inc_res_hs.add_halfspaces(ehs[np.newaxis, :], restart=True)
            total = np.vstack((halfspaces, extra_halfspaces[:i + 1, :]))
            hs = qhull.HalfspaceIntersection(total, feas_point)
            assert_allclose(inc_hs.halfspaces, inc_res_hs.halfspaces)
            assert_allclose(inc_hs.halfspaces, hs.halfspaces)
            assert_allclose(hs.intersections, inc_res_hs.intersections)
            self.assert_unordered_allclose(inc_hs.intersections, hs.intersections)
        inc_hs.close()

    def test_cube(self):
        halfspaces = np.array([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0], [0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
        point = np.array([0.5, 0.5, 0.5])
        hs = qhull.HalfspaceIntersection(halfspaces, point)
        qhalf_points = np.array([[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]])
        qhalf_facets = [[2, 4, 0], [4, 2, 1], [5, 2, 0], [2, 5, 1], [3, 4, 1], [4, 3, 0], [5, 3, 1], [3, 5, 0]]
        assert len(qhalf_facets) == len(hs.dual_facets)
        for a, b in zip(qhalf_facets, hs.dual_facets):
            assert set(a) == set(b)
        assert_allclose(hs.dual_points, qhalf_points)