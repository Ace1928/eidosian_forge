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
class TestNonFit:

    @classmethod
    def setup_class(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.endog = data.endog
        cls.exog = data.exog
        cls.ols_model = OLS(data.endog, data.exog)

    def test_df_resid(self):
        df_resid = self.endog.shape[0] - self.exog.shape[1]
        assert_equal(self.ols_model.df_resid, 9)