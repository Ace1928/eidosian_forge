import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def _compute_coordinate_deltas(self, x):
    x_coord, y_coord, z_coord = self._get_cordinates(x)
    dx = x_coord[:, None] - x_coord
    dy = y_coord[:, None] - y_coord
    dz = z_coord[:, None] - z_coord
    return (dx, dy, dz)