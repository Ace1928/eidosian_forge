import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4, concentrate_scale=False)
        params = np.r_[results_params['aust_ets1']['alpha'], results_params['aust_ets1']['beta'], results_params['aust_ets1']['gamma'], results_params['aust_ets1']['sigma2'], results_params['aust_ets1']['l0'], results_params['aust_ets1']['b0'], results_params['aust_ets1']['s0_0'], results_params['aust_ets1']['s0_1'], results_params['aust_ets1']['s0_2']]
        res = mod.filter(params)
        super().setup_class('aust_ets1', res)