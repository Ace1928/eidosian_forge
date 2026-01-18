import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
class TestUnconstrainStationaryUnivariate:
    cases = [(np.array([-2.0 / (1 + 2.0 ** 2) ** 0.5]), np.array([2.0]))]

    def test_cases(self):
        for constrained, unconstrained in self.cases:
            result = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(result, unconstrained)