import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestStationaryMultivariate:
    constrained_cases = [np.array([[0]]), np.array([[0.1]]), np.array([[-0.5]]), np.array([[0.999]]), [np.array([[0]])], np.array([[0.8, -0.2]]), [np.array([[0.8]]), np.array([[-0.2]])], [np.array([[0.3, 0.01], [-0.23, 0.15]]), np.array([[0.1, 0.03], [0.05, -0.3]])], np.array([[0.3, 0.01, 0.1, 0.03], [-0.23, 0.15, 0.05, -0.3]])]
    unconstrained_cases = [np.array([[0]]), np.array([[-40.42]]), np.array([[0.123]]), [np.array([[0]])], np.array([[100, 50]]), [np.array([[100]]), np.array([[50]])], [np.array([[30, 1], [-23, 15]]), np.array([[10, 0.3], [0.5, -30]])], np.array([[30, 1, 10, 0.3], [-23, 15, 0.5, -30]])]

    def test_cases(self):
        for constrained in self.constrained_cases:
            if type(constrained) is list:
                cov = np.eye(constrained[0].shape[0])
            else:
                cov = np.eye(constrained.shape[0])
            unconstrained, _ = tools.unconstrain_stationary_multivariate(constrained, cov)
            reconstrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            assert_allclose(reconstrained, constrained)
        for unconstrained in self.unconstrained_cases:
            if type(unconstrained) is list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            reunconstrained, _ = tools.unconstrain_stationary_multivariate(constrained, cov)
            assert_allclose(reunconstrained, unconstrained, atol=0.0001)