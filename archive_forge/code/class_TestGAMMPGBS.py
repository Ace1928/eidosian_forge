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
class TestGAMMPGBS(CheckGAMMixin):

    @classmethod
    def setup_class(cls):
        sp = np.array([0.830689464223685, 425.361212061649])
        cls.s_scale = s_scale = np.array([2.443955e-06, 0.007945455])
        x_spline = df_autos[['weight', 'hp']].values
        cls.exog = np.asarray(patsy.dmatrix('fuel + drive', data=df_autos))
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3], variable_names=['weight', 'hp'], constraints='center', include_intercept=True)
        alpha0 = 1 / s_scale * sp / 2
        gam_bs = GLMGam(df_autos['city_mpg'], exog=cls.exog, smoother=bs, alpha=alpha0.tolist())
        cls.res1a = gam_bs.fit(use_t=True)
        cls.res1b = gam_bs.fit(method='newton', use_t=True)
        cls.res1 = cls.res1a._results
        cls.res2 = results_mpg_bs.mpg_bs
        cls.rtol_fitted = 1e-08
        cls.covp_corrfact = 1
        cls.alpha = [169947.78222669504, 26767.58046340008]

    @classmethod
    def _init(cls):
        pass

    def test_edf(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.edf, res2.edf_all, rtol=1e-06)
        hat = res1.get_hat_matrix_diag()
        assert_allclose(hat, res2.hat, rtol=1e-06)

    def test_smooth(self):
        res1 = self.res1
        res2 = self.res2
        smoothers = res1.model.smoother.smoothers
        pen_matrix0 = smoothers[0].cov_der2
        assert_allclose(pen_matrix0, res2.smooth0.S * res2.smooth0.S_scale, rtol=1e-06)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        predicted = res1.predict(self.exog[2:4], res1.model.smoother.x[2:4])
        assert_allclose(predicted, res1.fittedvalues[2:4], rtol=1e-13)
        assert_allclose(predicted, res2.fitted_values[2:4], rtol=self.rtol_fitted)

    def test_crossval(self):
        mod = self.res1.model
        assert_equal(mod.alpha, self.alpha)
        assert_allclose(self.res1.scale, 4.706482135439112, rtol=1e-13)
        alpha_aic = mod.select_penweight()[0]
        assert_allclose(alpha_aic, [112487.81362014, 129.89155677], rtol=0.001)
        assert_equal(mod.alpha, self.alpha)
        assert_equal(mod.penal.start_idx, 4)
        pm = mod.penal.penalty_matrix()
        assert_equal(pm[:, :4], 0)
        assert_equal(pm[:4, :], 0)
        assert_allclose(self.res1.scale, 4.706482135439112, rtol=1e-13)
        np.random.seed(987125)
        alpha_cv, _ = mod.select_penweight_kfold(k_folds=3, k_grid=6)
        assert_allclose(alpha_cv, [10000000.0, 630.957344480193], rtol=1e-05)
        assert_equal(mod.alpha, self.alpha)
        assert_equal(mod.penal.start_idx, 4)
        pm = mod.penal.penalty_matrix()
        assert_equal(pm[:, :4], 0)
        assert_equal(pm[:4, :], 0)
        assert_allclose(self.res1.scale, 4.706482135439112, rtol=1e-13)