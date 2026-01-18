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
class TestVertexNeighborVertices:

    def _check(self, tri):
        expected = [set() for j in range(tri.points.shape[0])]
        for s in tri.simplices:
            for a in s:
                for b in s:
                    if a != b:
                        expected[a].add(b)
        indptr, indices = tri.vertex_neighbor_vertices
        got = [set(map(int, indices[indptr[j]:indptr[j + 1]])) for j in range(tri.points.shape[0])]
        assert_equal(got, expected, err_msg=f'{got!r} != {expected!r}')

    def test_triangle(self):
        points = np.array([(0, 0), (0, 1), (1, 0)], dtype=np.float64)
        tri = qhull.Delaunay(points)
        self._check(tri)

    def test_rectangle(self):
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
        tri = qhull.Delaunay(points)
        self._check(tri)

    def test_complicated(self):
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5), (0.9, 0.5)], dtype=np.float64)
        tri = qhull.Delaunay(points)
        self._check(tri)