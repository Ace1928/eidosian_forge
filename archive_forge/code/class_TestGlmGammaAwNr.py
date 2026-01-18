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
class TestGlmGammaAwNr(CheckWeight):

    @classmethod
    def setup_class(cls):
        from .results.results_glm import CancerLog
        res2 = CancerLog()
        endog = res2.endog
        exog = res2.exog[:, :-1]
        exog = sm.add_constant(exog, prepend=True)
        aweights = np.repeat(1, len(endog))
        aweights[::5] = 5
        aweights[::13] = 3
        model = sm.GLM(endog, exog, family=sm.families.Gamma(link=sm.families.links.Log()), var_weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=0)
        cls.res2 = res_r.results_gamma_aweights_nonrobust

    def test_r_llf(self):
        scale = self.res1.deviance / self.res1._iweights.sum()
        ll = self.res1.family.loglike(self.res1.model.endog, self.res1.mu, freq_weights=self.res1._var_weights, scale=scale)
        assert_allclose(ll, self.res2.ll, atol=1e-06, rtol=1e-07)