from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
class ExponentialFittingProblem:
    """Provide data and function for exponential fitting in the form
    y = a + exp(b * x) + noise."""

    def __init__(self, a, b, noise, n_outliers=1, x_range=(-1, 1), n_points=11, random_seed=None):
        np.random.seed(random_seed)
        self.m = n_points
        self.n = 2
        self.p0 = np.zeros(2)
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self.y = a + np.exp(b * self.x)
        self.y += noise * np.random.randn(self.m)
        outliers = np.random.randint(0, self.m, n_outliers)
        self.y[outliers] += 50 * noise * np.random.rand(n_outliers)
        self.p_opt = np.array([a, b])

    def fun(self, p):
        return p[0] + np.exp(p[1] * self.x) - self.y

    def jac(self, p):
        J = np.empty((self.m, self.n))
        J[:, 0] = 1
        J[:, 1] = self.x * np.exp(p[1] * self.x)
        return J