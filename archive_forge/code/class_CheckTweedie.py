import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
class CheckTweedie:

    def test_resid(self):
        idx1 = len(self.res1.resid_response) - 1
        idx2 = len(self.res2.resid_response) - 1
        assert_allclose(np.concatenate((self.res1.resid_response[:17], [self.res1.resid_response[idx1]])), np.concatenate((self.res2.resid_response[:17], [self.res2.resid_response[idx2]])), rtol=1e-05, atol=1e-05)
        assert_allclose(np.concatenate((self.res1.resid_pearson[:17], [self.res1.resid_pearson[idx1]])), np.concatenate((self.res2.resid_pearson[:17], [self.res2.resid_pearson[idx2]])), rtol=1e-05, atol=1e-05)
        assert_allclose(np.concatenate((self.res1.resid_deviance[:17], [self.res1.resid_deviance[idx1]])), np.concatenate((self.res2.resid_deviance[:17], [self.res2.resid_deviance[idx2]])), rtol=1e-05, atol=1e-05)
        assert_allclose(np.concatenate((self.res1.resid_working[:17], [self.res1.resid_working[idx1]])), np.concatenate((self.res2.resid_working[:17], [self.res2.resid_working[idx2]])), rtol=1e-05, atol=1e-05)

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-06, rtol=1000000.0)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-05, rtol=1e-05)

    def test_deviance(self):
        assert_allclose(self.res1.deviance, self.res2.deviance, atol=1e-06, rtol=1e-06)

    def test_df(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_fittedvalues(self):
        idx1 = len(self.res1.fittedvalues) - 1
        idx2 = len(self.res2.resid_response) - 1
        assert_allclose(np.concatenate((self.res1.fittedvalues[:17], [self.res1.fittedvalues[idx1]])), np.concatenate((self.res2.fittedvalues[:17], [self.res2.fittedvalues[idx2]])), atol=0.0001, rtol=0.0001)

    def test_summary(self):
        self.res1.summary()
        self.res1.summary2()