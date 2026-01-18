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
class TestGLMPoisson(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        np.random.seed(987125643)
        endog_count = np.random.poisson(endog)
        cls.cov_type = 'HC0'
        mod1 = GLM(endog_count, exog, family=families.Poisson())
        cls.res1 = mod1.fit(cov_type='HC0')
        mod1 = smd.Poisson(endog_count, exog)
        cls.res2 = mod1.fit(cov_type='HC0')
        cls.res1.rtol = 1e-11