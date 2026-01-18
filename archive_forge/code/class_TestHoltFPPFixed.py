import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltFPPFixed(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, concentrate_scale=False, initialization_method='simple')
        params = [results_params['air_fpp1']['alpha'], results_params['air_fpp1']['beta_star'], results_params['air_fpp1']['sigma2']]
        params[1] = params[0] * params[1]
        res = mod.filter(params)
        super().setup_class('air_fpp1', res)

    def test_conf_int(self):
        j = np.arange(1, 14)
        alpha, beta, sigma2 = self.res.params
        c = np.r_[0, alpha + beta * j]
        se = (sigma2 * (1 + np.cumsum(c ** 2))) ** 0.5
        assert_allclose(self.forecast.se_mean, se)