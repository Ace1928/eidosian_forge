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
class TestGAM5Bfgs(CheckGAMMixin):
    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        x = data_mcycle['times'].values
        endog = data_mcycle['accel']
        cc = CyclicCubicSplines(x, df=[6], constraints='center')
        gam_cc = GLMGam(endog, smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='bfgs')
        cls.res2 = results_pls.pls5
        cls.rtol_fitted = 1e-05
        cls.covp_corrfact = 1

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        predicted = res1.predict(None, res1.model.smoother.x[2:4])
        assert_allclose(predicted, res1.fittedvalues[2:4], rtol=1e-13)
        assert_allclose(predicted, res2.fitted_values[2:4], rtol=self.rtol_fitted)