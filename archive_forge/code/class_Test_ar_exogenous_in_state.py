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
class Test_ar_exogenous_in_state(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog)) ** 2
        kwargs['mle_regression'] = False
        super().setup_class(7, *args, **kwargs)
        cls.true_regression_coefficient = cls.true_params[0]
        cls.true_params = cls.true_params[1:]

    def test_loglike(self):
        self.result = self.model.filter(self.true_params)
        assert_allclose(self.result.llf, self.true_loglike, atol=2)

    def test_regression_coefficient(self):
        self.result = self.model.filter(self.true_params)
        assert_allclose(self.result.filter_results.filtered_state[3][-1], self.true_regression_coefficient, self.decimal)