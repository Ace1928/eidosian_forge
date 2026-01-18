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
class TestArrays2dEndog(TestArrays):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.endog = np.random.random((10, 1))
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)