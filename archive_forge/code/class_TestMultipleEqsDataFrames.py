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
class TestMultipleEqsDataFrames(TestDataFrames):

    @classmethod
    def setup_class(cls):
        cls.endog = endog = pd.DataFrame(np.random.random((10, 4)), columns=['y_1', 'y_2', 'y_3', 'y_4'])
        exog = pd.DataFrame(np.random.random((10, 2)), columns=['x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        neqs = 4
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input, index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input, index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input, index=exog.columns, columns=exog.columns)
        cls.cov_eq_input = np.random.random((neqs, neqs))
        cls.cov_eq_result = pd.DataFrame(cls.cov_eq_input, index=endog.columns, columns=endog.columns)
        cls.col_eq_input = np.random.random((nvars, neqs))
        cls.col_eq_result = pd.DataFrame(cls.col_eq_input, index=exog.columns, columns=endog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = ['y_1', 'y_2', 'y_3', 'y_4']
        cls.row_labels = cls.exog.index

    def test_attach(self):
        data = self.data
        assert_series_equal(data.wrap_output(self.col_input, 'columns'), self.col_result)
        assert_series_equal(data.wrap_output(self.row_input, 'rows'), self.row_result)
        assert_frame_equal(data.wrap_output(self.cov_input, 'cov'), self.cov_result)
        assert_frame_equal(data.wrap_output(self.cov_eq_input, 'cov_eq'), self.cov_eq_result)
        assert_frame_equal(data.wrap_output(self.col_eq_input, 'columns_eq'), self.col_eq_result)