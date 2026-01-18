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
class TestGLMPoissonHC1Fit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit(cov_type='HC1')
        cls.bse_rob = cls.res1.bse
        cls.corr_fact = cls.get_correction_factor(cls.res1, sub_kparams=False)