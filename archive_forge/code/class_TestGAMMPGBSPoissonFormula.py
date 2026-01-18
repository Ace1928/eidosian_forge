import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pandas as pd
import pytest
import patsy
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
from statsmodels.gam.smooth_basis import (BSplines, CyclicCubicSplines)
from statsmodels.gam.generalized_additive_model import (
from statsmodels.tools.linalg import matrix_sqrt, transf_constraints
from .results import results_pls, results_mpg_bs, results_mpg_bs_poisson
class TestGAMMPGBSPoissonFormula(TestGAMMPGBSPoisson):

    @classmethod
    def setup_class(cls):
        sp = np.array([40491.3940640059, 232455.530262537])
        cls.s_scale = s_scale = np.array([2.443955e-06, 0.007945455])
        cls.exog = patsy.dmatrix('fuel + drive', data=df_autos)
        x_spline = df_autos[['weight', 'hp']].values
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3], variable_names=['weight', 'hp'], constraints='center', include_intercept=True)
        alpha0 = 1 / s_scale * sp / 2
        gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', df_autos, smoother=bs, family=family.Poisson(), alpha=alpha0)
        cls.res1a = gam_bs.fit(use_t=False)
        cls.res1b = gam_bs.fit(method='newton', use_t=True)
        cls.res1 = cls.res1a._results
        cls.res2 = results_mpg_bs_poisson.mpg_bs_poisson
        cls.rtol_fitted = 1e-08
        cls.covp_corrfact = 1

    def test_names_wrapper(self):
        res1a = self.res1a
        xnames = ['Intercept', 'fuel[T.gas]', 'drive[T.fwd]', 'drive[T.rwd]', 'weight_s0', 'weight_s1', 'weight_s2', 'weight_s3', 'weight_s4', 'weight_s5', 'weight_s6', 'weight_s7', 'weight_s8', 'weight_s9', 'weight_s10', 'hp_s0', 'hp_s1', 'hp_s2', 'hp_s3', 'hp_s4', 'hp_s5', 'hp_s6', 'hp_s7', 'hp_s8']
        assert_equal(res1a.model.exog_names, xnames)
        assert_equal(res1a.model.design_info_linear.column_names, xnames[:4])
        assert_equal(res1a.fittedvalues.iloc[2:4].index.values, [2, 3])
        assert_equal(res1a.params.index.values, xnames)
        assert_(isinstance(res1a.params, pd.Series))
        assert_(isinstance(res1a, GLMGamResultsWrapper))
        assert_(isinstance(res1a._results, GLMGamResults))