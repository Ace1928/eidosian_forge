import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersDampedETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, damped_trend=True, seasonal=4, concentrate_scale=False)
        params = np.r_[results_params['aust_ets2']['alpha'], results_params['aust_ets2']['beta'], results_params['aust_ets2']['gamma'], results_params['aust_ets2']['phi'], results_params['aust_ets2']['sigma2'], results_params['aust_ets2']['l0'], results_params['aust_ets2']['b0'], results_params['aust_ets2']['s0_0'], results_params['aust_ets2']['s0_1'], results_params['aust_ets2']['s0_2']]
        res = mod.filter(params)
        super().setup_class('aust_ets2', res)

    def test_mle_estimates(self):
        mle_res = self.res.model.fit(disp=0, maxiter=100)
        assert_(self.res.llf <= mle_res.llf)