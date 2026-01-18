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
class TestArrays1dExog(TestArrays):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.endog = np.random.random(10)
        exog = np.random.random(10)
        cls.data = sm_data.handle_data(cls.endog, exog)
        cls.exog = exog[:, None]
        cls.xnames = ['x1']
        cls.ynames = 'y'

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog.squeeze())