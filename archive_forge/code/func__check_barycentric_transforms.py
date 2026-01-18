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
def _check_barycentric_transforms(self, tri, err_msg='', unit_cube=False, unit_cube_tol=0):
    """Check that a triangulation has reasonable barycentric transforms"""
    vertices = tri.points[tri.simplices]
    sc = 1 / (tri.ndim + 1.0)
    centroids = vertices.sum(axis=1) * sc

    def barycentric_transform(tr, x):
        r = tr[:, -1, :]
        Tinv = tr[:, :-1, :]
        return np.einsum('ijk,ik->ij', Tinv, x - r)
    eps = np.finfo(float).eps
    c = barycentric_transform(tri.transform, centroids)
    with np.errstate(invalid='ignore'):
        ok = np.isnan(c).all(axis=1) | (abs(c - sc) / sc < 0.1).all(axis=1)
    assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')
    q = vertices[:, :-1, :] - vertices[:, -1, None, :]
    volume = np.array([np.linalg.det(q[k, :, :]) for k in range(tri.nsimplex)])
    ok = np.isfinite(tri.transform[:, 0, 0]) | (volume < np.sqrt(eps))
    assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')
    j = tri.find_simplex(centroids)
    ok = (j != -1) | np.isnan(tri.transform[:, 0, 0])
    assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')
    if unit_cube:
        at_boundary = (centroids <= unit_cube_tol).any(axis=1)
        at_boundary |= (centroids >= 1 - unit_cube_tol).any(axis=1)
        ok = (j != -1) | at_boundary
        assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')