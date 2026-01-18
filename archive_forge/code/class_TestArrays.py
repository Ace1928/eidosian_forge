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
class TestArrays:

    @classmethod
    def setup_class(cls):
        cls.endog = np.random.random(10)
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_result = cls.col_input = np.random.random(nvars)
        cls.row_result = cls.row_input = np.random.random(nrows)
        cls.cov_result = cls.cov_input = np.random.random((nvars, nvars))
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y'
        cls.row_labels = None

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_attach(self):
        data = self.data
        np.testing.assert_equal(data.wrap_output(self.col_input, 'columns'), self.col_result)
        np.testing.assert_equal(data.wrap_output(self.row_input, 'rows'), self.row_result)
        np.testing.assert_equal(data.wrap_output(self.cov_input, 'cov'), self.cov_result)

    def test_names(self):
        data = self.data
        np.testing.assert_equal(data.xnames, self.xnames)
        np.testing.assert_equal(data.ynames, self.ynames)

    def test_labels(self):
        np.testing.assert_(np.all(self.data.row_labels == self.row_labels))