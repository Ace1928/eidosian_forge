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
class TestConvexHull:

    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.ConvexHull, masked_array)

    def test_array_with_nans_fails(self):
        points_with_nan = np.array([(0, 0), (1, 1), (2, np.nan)], dtype=np.float64)
        assert_raises(ValueError, qhull.ConvexHull, points_with_nan)

    @pytest.mark.parametrize('name', sorted(DATASETS))
    def test_hull_consistency_tri(self, name):
        points = DATASETS[name]
        tri = qhull.Delaunay(points)
        hull = qhull.ConvexHull(points)
        assert_hulls_equal(points, tri.convex_hull, hull.simplices)
        if points.shape[1] == 2:
            assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))
        else:
            assert_equal(np.unique(hull.simplices), hull.vertices)

    @pytest.mark.parametrize('name', sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        chunks, _ = INCREMENTAL_DATASETS[name]
        points = np.concatenate(chunks, axis=0)
        obj = qhull.ConvexHull(chunks[0], incremental=True)
        for chunk in chunks[1:]:
            obj.add_points(chunk)
        obj2 = qhull.ConvexHull(points)
        obj3 = qhull.ConvexHull(chunks[0], incremental=True)
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0), restart=True)
        assert_hulls_equal(points, obj.simplices, obj2.simplices)
        assert_hulls_equal(points, obj.simplices, obj3.simplices)

    def test_vertices_2d(self):
        np.random.seed(1234)
        points = np.random.rand(30, 2)
        hull = qhull.ConvexHull(points)
        assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))
        x, y = hull.points[hull.vertices].T
        angle = np.arctan2(y - y.mean(), x - x.mean())
        assert_(np.all(np.diff(np.unwrap(angle)) > 0))

    def test_volume_area(self):
        points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        tri = qhull.ConvexHull(points)
        assert_allclose(tri.volume, 1.0, rtol=1e-14)
        assert_allclose(tri.area, 6.0, rtol=1e-14)

    @pytest.mark.parametrize('incremental', [False, True])
    def test_good2d(self, incremental):
        points = np.array([[0.2, 0.2], [0.2, 0.4], [0.4, 0.4], [0.4, 0.2], [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points, incremental=incremental, qhull_options='QG4')
        expected = np.array([False, True, False, False], dtype=bool)
        actual = hull.good
        assert_equal(actual, expected)

    @pytest.mark.parametrize('visibility', ['QG4', 'QG-4'])
    @pytest.mark.parametrize('new_gen, expected', [(np.array([[0.3, 0.7]]), np.array([False, False, False, False, False], dtype=bool)), (np.array([[0.3, -0.7]]), np.array([False, True, False, False, False], dtype=bool)), (np.array([[0.3, 0.41]]), np.array([False, False, False, True, True], dtype=bool)), (np.array([[0.5, 0.6], [0.6, 0.6]]), np.array([False, False, True, False, False], dtype=bool)), (np.array([[0.3, 0.6 + 1e-16]]), np.array([False, False, False, False, False], dtype=bool))])
    def test_good2d_incremental_changes(self, new_gen, expected, visibility):
        points = np.array([[0.2, 0.2], [0.2, 0.4], [0.4, 0.4], [0.4, 0.2], [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points, incremental=True, qhull_options=visibility)
        hull.add_points(new_gen)
        actual = hull.good
        if '-' in visibility:
            expected = np.invert(expected)
        assert_equal(actual, expected)

    @pytest.mark.parametrize('incremental', [False, True])
    def test_good2d_no_option(self, incremental):
        points = np.array([[0.2, 0.2], [0.2, 0.4], [0.4, 0.4], [0.4, 0.2], [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points, incremental=incremental)
        actual = hull.good
        assert actual is None
        if incremental:
            hull.add_points(np.zeros((1, 2)))
            actual = hull.good
            assert actual is None

    @pytest.mark.parametrize('incremental', [False, True])
    def test_good2d_inside(self, incremental):
        points = np.array([[0.2, 0.2], [0.2, 0.4], [0.4, 0.4], [0.4, 0.2], [0.3, 0.3]])
        hull = qhull.ConvexHull(points=points, incremental=incremental, qhull_options='QG4')
        expected = np.array([False, False, False, False], dtype=bool)
        actual = hull.good
        assert_equal(actual, expected)

    @pytest.mark.parametrize('incremental', [False, True])
    def test_good3d(self, incremental):
        points = np.array([[0.0, 0.0, 0.0], [0.90029516, -0.39187448, 0.18948093], [0.4867642, -0.72627633, 0.48536925], [0.5765153, -0.81179274, -0.09285832], [0.67846893, -0.71119562, 0.1840671]])
        hull = qhull.ConvexHull(points=points, incremental=incremental, qhull_options='QG0')
        expected = np.array([True, False, False, False], dtype=bool)
        assert_equal(hull.good, expected)