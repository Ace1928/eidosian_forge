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
class TestADFNoConstant2(CheckADF):

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression='n', autolag=None, maxlag=1)
        cls.teststat = -2.4511596
        cls.pvalue = 0.013747
        cls.critvalues = [-2.587, -1.95, -1.617]
        _, _1, _2, cls.store = adfuller(cls.y, regression='n', autolag=None, maxlag=1, store=True)

    def test_store_str(self):
        assert_equal(self.store.__str__(), 'Augmented Dickey-Fuller Test Results')