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
class TestACFMissing(CheckCorrGram):

    @classmethod
    def setup_class(cls):
        cls.x = np.concatenate((np.array([np.nan]), cls.x))
        cls.acf = cls.results['acvar']
        cls.qstat = cls.results['Q1']
        cls.res_drop = acf(cls.x, nlags=40, qstat=True, alpha=0.05, missing='drop', fft=False)
        cls.res_conservative = acf(cls.x, nlags=40, qstat=True, alpha=0.05, fft=False, missing='conservative')
        cls.acf_none = np.empty(40) * np.nan
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = acf(cls.x, nlags=40, qstat=True, alpha=0.05, missing='none', fft=False)

    def test_raise(self):
        with pytest.raises(MissingDataError):
            acf(self.x, nlags=40, qstat=True, fft=False, alpha=0.05, missing='raise')

    def test_acf_none(self):
        assert_almost_equal(self.res_none[0][1:41], self.acf_none, DECIMAL_8)

    def test_acf_drop(self):
        assert_almost_equal(self.res_drop[0][1:41], self.acf, DECIMAL_8)

    def test_acf_conservative(self):
        assert_almost_equal(self.res_conservative[0][1:41], self.acf, DECIMAL_8)

    def test_qstat_none(self):
        assert_almost_equal(self.res_none[2], self.qstat_none, DECIMAL_3)