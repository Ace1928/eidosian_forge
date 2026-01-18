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
class TestGlmGaussianWLS(CheckWeight):

    @classmethod
    def setup_class(cls):
        import statsmodels.formula.api as smf
        data = sm.datasets.cpunish.load_pandas()
        endog = data.endog
        data = data.exog
        data['EXECUTIONS'] = endog
        data['INCOME'] /= 1000
        aweights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        model = smf.glm('EXECUTIONS ~ INCOME + SOUTH - 1', data=data, family=sm.families.Gaussian(link=sm.families.links.Identity()), var_weights=aweights)
        wlsmodel = smf.wls('EXECUTIONS ~ INCOME + SOUTH - 1', data=data, weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=1e-25)
        cls.res2 = wlsmodel.fit()