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
class TestGlmGaussianAwNr(CheckWeight):

    @classmethod
    def setup_class(cls):
        import statsmodels.formula.api as smf
        data = sm.datasets.cpunish.load_pandas()
        endog = data.endog
        data = data.exog
        data['EXECUTIONS'] = endog
        data['INCOME'] /= 1000
        aweights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        model = smf.glm('EXECUTIONS ~ INCOME + SOUTH - 1', data=data, family=sm.families.Gaussian(link=sm.families.links.Log()), var_weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=0)
        cls.res2 = res_r.results_gaussian_aweights_nonrobust

    def test_r_llf(self):
        res1 = self.res1
        res2 = self.res2
        model = self.res1.model
        scale = res1.scale * model.df_resid / model.wnobs
        wts = model.freq_weights
        llf = model.family.loglike(model.endog, res1.mu, freq_weights=wts, scale=scale)
        adj_sm = -1 / 2 * ((model.endog - res1.mu) ** 2).sum() / scale
        adj_r = -model.wnobs / 2 + np.sum(np.log(model.var_weights)) / 2
        llf_adj = llf - adj_sm + adj_r
        assert_allclose(llf_adj, res2.ll, atol=1e-06, rtol=1e-07)