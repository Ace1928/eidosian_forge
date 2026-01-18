import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.discrete.discrete_model as smd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.regression.linear_model import OLS
from statsmodels.base.covtype import get_robustcov_results
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from numpy.testing import assert_allclose, assert_equal, assert_
import statsmodels.tools._testing as smt
from .results import results_count_robust_cluster as results_st
class TestGLMGaussHACPanel(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'hac-panel'
        time = np.tile(np.arange(7), 5)[:-1]
        mod1 = GLM(endog.copy(), exog.copy(), family=families.Gaussian())
        kwds = dict(time=time, maxlags=2, kernel=sw.weights_uniform, use_correction='hac', df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-panel', cov_kwds=kwds)
        cls.res1b = mod1.fit(cov_type='nw-panel', cov_kwds=kwds)
        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-panel', cov_kwds=kwds)

    def test_kwd(self):
        assert_allclose(self.res1b.bse, self.res1.bse, rtol=1e-12)