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
class TestGLMLogitOffset(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        endog_bin = (endog > endog.mean()).astype(int)
        cls.cov_type = 'cluster'
        offset = np.ones(endog_bin.shape[0])
        mod1 = GLM(endog_bin, exog, family=families.Binomial(), offset=offset)
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
        mod1 = smd.Logit(endog_bin, exog, offset=offset)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))