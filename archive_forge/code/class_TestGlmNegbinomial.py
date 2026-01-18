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
class TestGlmNegbinomial(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        """
        Test Negative Binomial family with log link
        """
        cls.decimal_resid = DECIMAL_1
        cls.decimal_params = DECIMAL_3
        cls.decimal_resids = -1
        cls.decimal_fittedvalues = DECIMAL_1
        from statsmodels.datasets.committee import load
        cls.data = load()
        cls.data.endog = np.require(cls.data.endog, requirements='W')
        cls.data.exog = np.require(cls.data.exog, requirements='W')
        cls.data.exog[:, 2] = np.log(cls.data.exog[:, 2])
        interaction = cls.data.exog[:, 2] * cls.data.exog[:, 1]
        cls.data.exog = np.column_stack((cls.data.exog, interaction))
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DomainWarning)
            with pytest.warns(UserWarning):
                fam = sm.families.NegativeBinomial()
        cls.res1 = GLM(cls.data.endog, cls.data.exog, family=fam).fit(scale='x2')
        from .results.results_glm import Committee
        res2 = Committee()
        res2.aic_R += 2
        cls.res2 = res2
        cls.has_edispersion = True