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
class Airline(SARIMAXStataTests):
    """
    Multiplicative SARIMA model: "Airline" model

    Stata arima documentation, Example 3
    """

    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = np.log(true['data'])
        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)
        cls.model = sarimax.SARIMAX(endog, *args, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend='n', **kwargs)
        params = np.r_[true['params_ma'], true['params_seasonal_ma'], true['params_variance']]
        cls.result = cls.model.filter(params)

    def test_mle(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = self.model.fit(disp=-1)
            assert_allclose(result.params, self.result.params, atol=0.0001)