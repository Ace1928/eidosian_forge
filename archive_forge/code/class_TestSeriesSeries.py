from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
class TestSeriesSeries(TestDataFrames):

    @classmethod
    def setup_class(cls):
        cls.endog = pd.Series(np.random.random(10), name='y_1')
        exog = pd.Series(np.random.random(10), name='x_1')
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 1
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input, index=[exog.name])
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input, index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input, index=[exog.name], columns=[exog.name])
        cls.xnames = ['x_1']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        assert_series_equal(self.data.orig_endog, self.endog)
        assert_series_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog.values[:, None])