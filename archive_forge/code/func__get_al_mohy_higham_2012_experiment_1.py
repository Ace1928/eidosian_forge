import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def _get_al_mohy_higham_2012_experiment_1():
    """
    Return the test matrix from Experiment (1) of [1]_.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    """
    A = np.array([[0.32346, 30000.0, 30000.0, 30000.0], [0, 0.30089, 30000.0, 30000.0], [0, 0, 0.3221, 30000.0], [0, 0, 0, 0.30744]], dtype=float)
    return A