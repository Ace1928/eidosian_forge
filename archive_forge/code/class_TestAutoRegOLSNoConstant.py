from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
class TestAutoRegOLSNoConstant(CheckAutoRegMixin):
    """f
    Test AR fit by OLS without a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sunspots.load()
        cls.res1 = AutoReg(np.asarray(data.endog), lags=9, trend='n').fit()
        cls.res2 = results_ar.ARResultsOLS(constant=False)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params)[model.hold_back:], self.res2.FVOLSnneg1start0, DECIMAL_4)
        assert_almost_equal(model.predict(params)[model.hold_back:], self.res2.FVOLSnneg1start9, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100), self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200), self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params)[model.hold_back:], self.res2.FVOLSdefault, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400), self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=424), self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=310), self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=316), self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=327), self.res2.FVOLSn15start312, DECIMAL_4)