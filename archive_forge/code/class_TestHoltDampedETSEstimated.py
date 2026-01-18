import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltDampedETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True, concentrate_scale=False)
        params = [results_params['air_ets']['alpha'], results_params['air_ets']['beta'], results_params['air_ets']['phi'], results_params['air_ets']['sigma2'], results_params['air_ets']['l0'], results_params['air_ets']['b0']]
        res = mod.filter(params)
        super().setup_class('air_ets', res)

    def test_mle_estimates(self):
        mle_res = self.res.model.fit(disp=0)
        assert_(self.res.llf <= mle_res.llf)