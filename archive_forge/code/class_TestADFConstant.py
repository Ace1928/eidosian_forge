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
class TestADFConstant(CheckADF):
    """
    Dickey-Fuller test for unit root
    """

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression='c', autolag=None, maxlag=4)
        cls.teststat = 0.97505319
        cls.pvalue = 0.99399563
        cls.critvalues = [-3.476, -2.883, -2.573]