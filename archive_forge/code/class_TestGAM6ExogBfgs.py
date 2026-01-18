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
class TestGAM6ExogBfgs:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        nobs = data_mcycle['times'].shape[0]
        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6], constraints='center')
        gam_cc = GLMGam(data_mcycle['accel'], np.ones(nobs), smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='bfgs')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-05
        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues, rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues, rtol=self.rtol_fitted)

    def test_exog(self):
        exog = self.res1.model.exog
        assert_allclose(exog[:10], pls6_exog, rtol=1e-13)