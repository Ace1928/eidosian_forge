import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def _vandermonde(x, degree):
    powers = _monomial_powers(x.shape[1], degree)
    return _rbfinterp_pythran._polynomial_matrix(x, powers)