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
class Test_seasonal_arma_trend_polynomial(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        kwargs['trend'] = [1, 0, 0, 1]
        kwargs['decimal'] = 3
        super().setup_class(44, *args, **kwargs)
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]

    def test_results(self):
        self.result = self.model.filter(self.true_params)
        self.result.summary()
        self.result.cov_params_default
        self.result.cov_params_oim
        self.result.cov_params_opg