from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestDataDimensions(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.endog_n_ = np.random.uniform(0, 20, size=30)
        cls.endog_n_one = cls.endog_n_[:, None]
        cls.exog_n_ = np.random.uniform(0, 20, size=30)
        cls.exog_n_one = cls.exog_n_[:, None]
        cls.degen_exog = cls.exog_n_one[:-1]
        cls.mod1 = OLS(cls.endog_n_one, cls.exog_n_one)
        cls.mod1.df_model += 1
        cls.res1 = cls.mod1.fit()
        cls.mod2 = OLS(cls.endog_n_one, cls.exog_n_one)
        cls.mod2.df_model += 1
        cls.res2 = cls.mod2.fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL_4)