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
class TestVoronoi:

    @pytest.mark.parametrize('qhull_opts, extra_pts', [('Qbb Qc Qz', 1), ('Qbb Qc', 0)])
    @pytest.mark.parametrize('n_pts', [50, 100])
    @pytest.mark.parametrize('ndim', [2, 3])
    def test_point_region_structure(self, qhull_opts, n_pts, extra_pts, ndim):
        rng = np.random.default_rng(7790)
        points = rng.random((n_pts, ndim))
        vor = Voronoi(points, qhull_options=qhull_opts)
        pt_region = vor.point_region
        assert pt_region.max() == n_pts - 1 + extra_pts
        assert pt_region.size == len(vor.regions) - extra_pts
        assert len(vor.regions) == n_pts + extra_pts
        assert vor.points.shape[0] == n_pts
        if extra_pts:
            sublens = [len(x) for x in vor.regions]
            assert sublens.count(0) == 1
            assert sublens.index(0) not in pt_region

    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.Voronoi, masked_array)

    def test_simple(self):
        points = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        output = '\n        2\n        5 10 1\n        -10.101 -10.101\n           0.5    0.5\n           0.5    1.5\n           1.5    0.5\n           1.5    1.5\n        2 0 1\n        3 2 0 1\n        2 0 2\n        3 3 0 1\n        4 1 2 4 3\n        3 4 0 2\n        2 0 3\n        3 4 0 3\n        2 0 4\n        0\n        12\n        4 0 3 0 1\n        4 0 1 0 1\n        4 1 4 1 2\n        4 1 2 0 2\n        4 2 5 0 2\n        4 3 4 1 3\n        4 3 6 0 3\n        4 4 5 2 4\n        4 4 7 3 4\n        4 5 8 0 4\n        4 6 7 0 3\n        4 7 8 0 4\n        '
        self._compare_qvoronoi(points, output)

    def _compare_qvoronoi(self, points, output, **kw):
        """Compare to output from 'qvoronoi o Fv < data' to Voronoi()"""
        output = [list(map(float, x.split())) for x in output.strip().splitlines()]
        nvertex = int(output[1][0])
        vertices = list(map(tuple, output[3:2 + nvertex]))
        nregion = int(output[1][1])
        regions = [[int(y) - 1 for y in x[1:]] for x in output[2 + nvertex:2 + nvertex + nregion]]
        ridge_points = [[int(y) for y in x[1:3]] for x in output[3 + nvertex + nregion:]]
        ridge_vertices = [[int(y) - 1 for y in x[3:]] for x in output[3 + nvertex + nregion:]]
        vor = qhull.Voronoi(points, **kw)

        def sorttuple(x):
            return tuple(sorted(x))
        assert_allclose(vor.vertices, vertices)
        assert_equal(set(map(tuple, vor.regions)), set(map(tuple, regions)))
        p1 = list(zip(list(map(sorttuple, ridge_points)), list(map(sorttuple, ridge_vertices))))
        p2 = list(zip(list(map(sorttuple, vor.ridge_points.tolist())), list(map(sorttuple, vor.ridge_vertices))))
        p1.sort()
        p2.sort()
        assert_equal(p1, p2)

    @pytest.mark.parametrize('name', sorted(DATASETS))
    def test_ridges(self, name):
        points = DATASETS[name]
        tree = KDTree(points)
        vor = qhull.Voronoi(points)
        for p, v in vor.ridge_dict.items():
            if not np.all(np.asarray(v) >= 0):
                continue
            ridge_midpoint = vor.vertices[v].mean(axis=0)
            d = 1e-06 * (points[p[0]] - ridge_midpoint)
            dist, k = tree.query(ridge_midpoint + d, k=1)
            assert_equal(k, p[0])
            dist, k = tree.query(ridge_midpoint - d, k=1)
            assert_equal(k, p[1])

    def test_furthest_site(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
        output = '\n        2\n        3 5 1\n        -10.101 -10.101\n        0.6000000000000001    0.5\n           0.5 0.6000000000000001\n        3 0 2 1\n        2 0 1\n        2 0 2\n        0\n        3 0 2 1\n        5\n        4 0 2 0 2\n        4 0 4 1 2\n        4 0 1 0 1\n        4 1 4 0 1\n        4 2 4 0 2\n        '
        self._compare_qvoronoi(points, output, furthest_site=True)

    def test_furthest_site_flag(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
        vor = Voronoi(points)
        assert_equal(vor.furthest_site, False)
        vor = Voronoi(points, furthest_site=True)
        assert_equal(vor.furthest_site, True)

    @pytest.mark.parametrize('name', sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        if INCREMENTAL_DATASETS[name][0][0].shape[1] > 3:
            return
        chunks, opts = INCREMENTAL_DATASETS[name]
        points = np.concatenate(chunks, axis=0)
        obj = qhull.Voronoi(chunks[0], incremental=True, qhull_options=opts)
        for chunk in chunks[1:]:
            obj.add_points(chunk)
        obj2 = qhull.Voronoi(points)
        obj3 = qhull.Voronoi(chunks[0], incremental=True, qhull_options=opts)
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0), restart=True)
        assert_equal(len(obj.point_region), len(obj2.point_region))
        assert_equal(len(obj.point_region), len(obj3.point_region))
        for objx in (obj, obj3):
            vertex_map = {-1: -1}
            for i, v in enumerate(objx.vertices):
                for j, v2 in enumerate(obj2.vertices):
                    if np.allclose(v, v2):
                        vertex_map[i] = j

            def remap(x):
                if hasattr(x, '__len__'):
                    return tuple({remap(y) for y in x})
                try:
                    return vertex_map[x]
                except KeyError as e:
                    message = f'incremental result has spurious vertex at {objx.vertices[x]!r}'
                    raise AssertionError(message) from e

            def simplified(x):
                items = set(map(sorted_tuple, x))
                if () in items:
                    items.remove(())
                items = [x for x in items if len(x) > 1]
                items.sort()
                return items
            assert_equal(simplified(remap(objx.regions)), simplified(obj2.regions))
            assert_equal(simplified(remap(objx.ridge_vertices)), simplified(obj2.ridge_vertices))