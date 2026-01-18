import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
class TestBinomialVsVarWeights(CheckWeight):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = np.require(data.exog, requirements='W')
        data.endog = np.require(data.endog, requirements='W')
        data.exog /= data.exog.std(0)
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Binomial()).fit()
        weights = data.endog.sum(axis=1)
        endog2 = data.endog[:, 0] / weights
        cls.res2 = GLM(endog2, data.exog, family=sm.families.Binomial(), var_weights=weights).fit()