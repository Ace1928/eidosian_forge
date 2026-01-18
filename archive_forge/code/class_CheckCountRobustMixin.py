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
class CheckCountRobustMixin:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        if len(res1.params) == len(res2.params) - 1:
            mask = np.ones(len(res2.params), np.bool_)
            mask[-2] = False
            res2_params = res2.params[mask]
            res2_bse = res2.bse[mask]
        else:
            res2_params = res2.params
            res2_bse = res2.bse
        assert_allclose(res1._results.params, res2_params, 0.0001)
        assert_allclose(self.bse_rob / self.corr_fact, res2_bse, 6e-05)

    @classmethod
    def get_robust_clu(cls):
        res1 = cls.res1
        cov_clu = sw.cov_cluster(res1, group)
        cls.bse_rob = sw.se_cov(cov_clu)
        cls.corr_fact = cls.get_correction_factor(res1)

    @classmethod
    def get_correction_factor(cls, results, sub_kparams=True):
        mod = results.model
        nobs, k_vars = mod.exog.shape
        if sub_kparams:
            k_params = len(results.params)
        else:
            k_params = 0
        corr_fact = (nobs - 1.0) / float(nobs - k_params)
        return np.sqrt(corr_fact)

    def test_oth(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1._results.llf, res2.ll, 0.0001)
        assert_allclose(res1._results.llnull, res2.ll_0, 0.0001)

    def test_ttest(self):
        smt.check_ttest_tvalues(self.res1)

    def test_waldtest(self):
        smt.check_ftest_pvalues(self.res1)