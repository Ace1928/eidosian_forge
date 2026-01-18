import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestStationaryUnivariate:
    constrained_cases = [np.array([0]), np.array([0.1]), np.array([-0.5]), np.array([0.999])]
    unconstrained_cases = [np.array([10.0]), np.array([-40.42]), np.array([0.123])]

    def test_cases(self):
        for constrained in self.constrained_cases:
            unconstrained = tools.unconstrain_stationary_univariate(constrained)
            reconstrained = tools.constrain_stationary_univariate(unconstrained)
            assert_allclose(reconstrained, constrained)
        for unconstrained in self.unconstrained_cases:
            constrained = tools.constrain_stationary_univariate(unconstrained)
            reunconstrained = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(reunconstrained, unconstrained)