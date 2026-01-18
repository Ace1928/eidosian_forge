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
class TestWLS_CornerCases:

    @classmethod
    def setup_class(cls):
        cls.exog = np.ones((1,))
        cls.endog = np.ones((1,))
        weights = 1
        cls.wls_res = WLS(cls.endog, cls.exog, weights=weights).fit()

    def test_wrong_size_weights(self):
        weights = np.ones((10, 10))
        assert_raises(ValueError, WLS, self.endog, self.exog, weights=weights)