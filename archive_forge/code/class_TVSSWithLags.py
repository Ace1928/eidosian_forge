import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
class TVSSWithLags(TVSS):

    def __init__(self, endog):
        super().__init__(endog, _k_states=8)
        self['transition', 2:, :6] = np.eye(6)[..., None]
        self.ssm.initialize_approximate_diffuse(0.0001)