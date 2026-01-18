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
class Test_arma_exog_trend_polynomial_missing(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        endog = np.r_[results_sarimax.wpi1_data]
        kwargs['exog'] = ((endog - np.floor(endog)) ** 2)[1:]
        endog[9:19] = np.nan
        endog = endog[1:] - endog[:-1]
        endog[9] = np.nan
        kwargs['order'] = (3, 0, 2)
        kwargs['trend'] = [0, 0, 0, 1]
        kwargs['decimal'] = 1
        super().setup_class(52, *args, endog=endog, **kwargs)
        tps = cls.true_params
        cls.true_params[0] = (1 - tps[2:5].sum()) * tps[0]