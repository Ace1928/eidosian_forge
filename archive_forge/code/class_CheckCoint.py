from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
class CheckCoint:
    """
    Test Cointegration Test Results for 2-variable system

    Test values taken from Stata
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load_pandas()
    y1 = data.data['realcons'].values
    y2 = data.data['realgdp'].values

    def test_tstat(self):
        assert_almost_equal(self.coint_t, self.teststat, DECIMAL_4)