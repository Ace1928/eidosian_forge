import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestSESETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method='estimated', concentrate_scale=False)
        res = mod.filter([results_params['oil_ets']['alpha'], results_params['oil_ets']['sigma2'], results_params['oil_ets']['l0']])
        super().setup_class('oil_ets', res)

    def test_mle_estimates(self):
        mle_res = self.res.model.fit(disp=0)
        assert_(self.res.llf <= mle_res.llf)