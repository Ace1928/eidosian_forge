import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def _generate_icosahedron():
    x = np.array([[0, -1, -phi], [0, -1, +phi], [0, +1, -phi], [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])