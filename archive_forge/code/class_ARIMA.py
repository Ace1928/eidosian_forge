import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
class ARIMA(SARIMAXStataTests):
    """
    ARIMA model

    Stata arima documentation, Example 1
    """

    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = true['data']
        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)
        cls.model = sarimax.SARIMAX(endog, *args, order=(1, 1, 1), trend='c', **kwargs)
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'], true['params_variance']]
        cls.result = cls.model.filter(params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(result.params, self.result.params, atol=0.001)