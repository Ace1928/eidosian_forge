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
class TestFriedmanPredict(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, prediction

    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This follows the given Stata example, although it is not truly forecasting
    because it compares using the actual data (which is available in the
    example but just not used in the parameter MLE estimation) against dynamic
    prediction of that data. Here `test_predict` matches the first case, and
    `test_dynamic_predict` matches the second.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(results_sarimax.friedman2_predict)

    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_predict(self):
        assert_almost_equal(self.result.predict(), self.true['predict'], 3)

    def test_dynamic_predict(self):
        dynamic = len(self.true['data']['consump']) - 15 - 1
        assert_almost_equal(self.result.predict(dynamic=dynamic), self.true['dynamic_predict'], 3)