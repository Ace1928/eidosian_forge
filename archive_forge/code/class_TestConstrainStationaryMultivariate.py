import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestConstrainStationaryMultivariate:
    cases = [(np.array([[2.0]]), np.eye(1), np.array([[2.0 / (1 + 2.0 ** 2) ** 0.5]])), ([np.array([[2.0]])], np.eye(1), [np.array([[2.0 / (1 + 2.0 ** 2) ** 0.5]])])]
    eigval_cases = [[np.array([[0]])], [np.array([[100]]), np.array([[50]])], [np.array([[30, 1], [-23, 15]]), np.array([[10, 0.3], [0.5, -30]])]]

    def test_cases(self):
        for unconstrained, error_variance, constrained in self.cases:
            result = tools.constrain_stationary_multivariate(unconstrained, error_variance)
            assert_allclose(result[0], constrained)
        for unconstrained in self.eigval_cases:
            if type(unconstrained) is list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            companion = tools.companion_matrix([1] + [-np.squeeze(constrained[i]) for i in range(len(constrained))]).T
            assert_array_less(np.abs(np.linalg.eigvals(companion)), 1)