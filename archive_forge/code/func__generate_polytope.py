import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def _generate_polytope(name):
    polygons = ['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'nonagon', 'decagon', 'undecagon', 'dodecagon']
    polyhedra = ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']
    if name not in polygons and name not in polyhedra:
        raise ValueError('unrecognized polytope')
    if name in polygons:
        n = polygons.index(name) + 3
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        p = np.vstack([np.cos(thetas), np.sin(thetas)]).T
    elif name == 'tetrahedron':
        p = _generate_tetrahedron()
    elif name == 'cube':
        p = _generate_cube()
    elif name == 'octahedron':
        p = _generate_octahedron()
    elif name == 'dodecahedron':
        p = _generate_dodecahedron()
    elif name == 'icosahedron':
        p = _generate_icosahedron()
    return p / np.linalg.norm(p, axis=1, keepdims=True)