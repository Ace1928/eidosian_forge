import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def _sample_2d_data(self):
    x = np.array([0.5, 2.0, 3.0, 4.0, 5.5, 6.0])
    y = np.array([0.5, 2.0, 3.0, 4.0, 5.5, 6.0])
    z = np.array([[1, 2, 1, 2, 1, 1], [1, 2, 1, 2, 1, 1], [1, 2, 3, 2, 1, 1], [1, 2, 2, 2, 1, 1], [1, 2, 1, 2, 1, 1], [1, 2, 2, 2, 1, 1]])
    return (x, y, z)