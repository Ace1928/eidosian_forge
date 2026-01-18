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
def _help_expm_cond_search(A, A_norm, X, X_norm, eps, p):
    p = np.reshape(p, A.shape)
    p_norm = norm(p)
    perturbation = eps * p * (A_norm / p_norm)
    X_prime = expm(A + perturbation)
    scaled_relative_error = norm(X_prime - X) / (X_norm * eps)
    return -scaled_relative_error