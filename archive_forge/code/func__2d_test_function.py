import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def _2d_test_function(x):
    x1, x2 = (x[:, 0], x[:, 1])
    term1 = 0.75 * np.exp(-(9 * x1 - 2) ** 2 / 4 - (9 * x2 - 2) ** 2 / 4)
    term2 = 0.75 * np.exp(-(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) / 10)
    term3 = 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4 - (9 * x2 - 3) ** 2 / 4)
    term4 = -0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
    y = term1 + term2 + term3 + term4
    return y