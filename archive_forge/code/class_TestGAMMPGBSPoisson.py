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
class TestGAMMPGBSPoisson(CheckGAMMixin):

    @classmethod
    def setup_class(cls):
        sp = np.array([40491.3940640059, 232455.530262537])
        cls.s_scale = s_scale = np.array([2.443955e-06, 0.007945455])
        x_spline = df_autos[['weight', 'hp']].values
        cls.exog = patsy.dmatrix('fuel + drive', data=df_autos)
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3], variable_names=['weight', 'hp'], constraints='center', include_intercept=True)
        alpha0 = 1 / s_scale * sp / 2
        gam_bs = GLMGam(df_autos['city_mpg'], exog=cls.exog, smoother=bs, family=family.Poisson(), alpha=alpha0)
        xnames = cls.exog.design_info.column_names + gam_bs.smoother.col_names
        gam_bs.exog_names[:] = xnames
        cls.res1a = gam_bs.fit(use_t=False)
        cls.res1b = gam_bs.fit(method='newton', use_t=True)
        cls.res1 = cls.res1a._results
        cls.res2 = results_mpg_bs_poisson.mpg_bs_poisson
        cls.rtol_fitted = 1e-08
        cls.covp_corrfact = 1

    @classmethod
    def _init(cls):
        pass

    def test_edf(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.edf, res2.edf_all, rtol=1e-06)
        hat = res1.get_hat_matrix_diag()
        assert_allclose(hat, res2.hat, rtol=1e-06)
        assert_allclose(res1.aic, res2.aic, rtol=1e-08)
        assert_allclose(res1.deviance, res2.deviance, rtol=1e-08)
        assert_allclose(res1.df_resid, res2.residual_df, rtol=1e-08)

    def test_smooth(self):
        res1 = self.res1
        res2 = self.res2
        smoothers = res1.model.smoother.smoothers
        pen_matrix0 = smoothers[0].cov_der2
        assert_allclose(pen_matrix0, res2.smooth0.S * res2.smooth0.S_scale, rtol=1e-06)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        predicted = res1.predict(df_autos.iloc[2:4], res1.model.smoother.x[2:4])
        assert_allclose(predicted, res1.fittedvalues[2:4], rtol=1e-13)
        assert_allclose(predicted, res2.fitted_values[2:4], rtol=self.rtol_fitted)
        xp = pd.DataFrame(res1.model.smoother.x[2:4])
        linpred = res1.predict(df_autos.iloc[2:4], xp, which='linear')
        assert_allclose(linpred, res2.linear_predictors[2:4], rtol=self.rtol_fitted)
        assert_equal(predicted.index.values, [2, 3])
        assert_equal(linpred.index.values, [2, 3])

    def test_wald(self):
        res1 = self.res1
        res2 = self.res2
        wtt = res1.wald_test_terms(skip_single=True, combine_terms=['fuel', 'drive', 'weight', 'hp'], scalar=True)
        assert_allclose(wtt.statistic[:2], res2.pTerms_chi_sq, rtol=1e-07)
        assert_allclose(wtt.pvalues[:2], res2.pTerms_pv, rtol=1e-06)
        assert_equal(wtt.df_constraints[:2], res2.pTerms_df)

    def test_select_alpha(self):
        res1 = self.res1
        alpha_mgcv = res1.model.alpha
        res_s = res1.model.select_penweight()
        assert_allclose(res_s[0], alpha_mgcv, rtol=5e-05)