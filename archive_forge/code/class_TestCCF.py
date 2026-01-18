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
class TestCCF:
    """
    Test cross-correlation function
    """
    data = macrodata.load_pandas()
    x = data.data['unemp'].diff().dropna()
    y = data.data['infl'].diff().dropna()
    filename = os.path.join(CURR_DIR, 'results', 'results_ccf.csv')
    results = pd.read_csv(filename, delimiter=',')
    nlags = 20

    @classmethod
    def setup_class(cls):
        cls.ccf = cls.results['ccf']
        cls.res1 = ccf(cls.x, cls.y, nlags=cls.nlags, adjusted=False, fft=False)

    def test_ccf(self):
        assert_almost_equal(self.res1, self.ccf, DECIMAL_8)

    def test_confint(self):
        alpha = 0.05
        res2, confint = ccf(self.x, self.y, nlags=self.nlags, adjusted=False, fft=False, alpha=alpha)
        assert_equal(res2, self.res1)
        assert_almost_equal(res2 - confint[:, 0], confint[:, 1] - res2, DECIMAL_8)
        alpha1 = stats.norm.cdf(confint[:, 1] - res2, scale=1.0 / np.sqrt(len(self.x)))
        assert_almost_equal(alpha1, np.repeat(1 - alpha / 2.0, self.nlags), DECIMAL_8)