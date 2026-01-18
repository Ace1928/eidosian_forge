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