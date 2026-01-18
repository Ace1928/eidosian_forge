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
class TestGlmTweedieAwNr(CheckWeight):

    @classmethod
    def setup_class(cls):
        import statsmodels.formula.api as smf
        data = sm.datasets.fair.load_pandas()
        endog = data.endog
        data = data.exog
        data['fair'] = endog
        aweights = np.repeat(1, len(data.index))
        aweights[::5] = 5
        aweights[::13] = 3
        model = smf.glm('fair ~ age + yrs_married', data=data, family=sm.families.Tweedie(var_power=1.55, link=sm.families.links.Log()), var_weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=0)
        cls.res2 = res_r.results_tweedie_aweights_nonrobust