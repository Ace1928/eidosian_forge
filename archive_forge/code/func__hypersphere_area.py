import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def _hypersphere_area(dim, radius):
    return 2 * np.pi ** (dim / 2) / gamma(dim / 2) * radius ** (dim - 1)