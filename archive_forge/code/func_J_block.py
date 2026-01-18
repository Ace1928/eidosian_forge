import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def J_block(h, p):
    return np.array([[h ** 2 * p ** 2 / 12 - 1, -0.5 * h, -h ** 2 * p ** 2 / 12 + 1, -0.5 * h], [0.5 * h * p ** 2, h ** 2 * p ** 2 / 12 - 1, 0.5 * h * p ** 2, 1 - h ** 2 * p ** 2 / 12]])