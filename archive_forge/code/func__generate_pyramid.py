import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
def _generate_pyramid(n, axis):
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    P = np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
    P = np.concatenate((P, [[1, 0, 0]]))
    return np.roll(P, axis, axis=1)