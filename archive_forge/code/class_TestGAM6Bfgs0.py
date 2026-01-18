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
class TestGAM6Bfgs0:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6])
        gam_cc = GLMGam(data_mcycle['accel'], smoother=cc, alpha=0)
        cls.res1 = gam_cc.fit(method='bfgs')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-05
        pls6_fittedvalues = np.array([2.63203377595747, 3.41285892739456, 5.78168657308338, 7.35344779586831, 8.89178704316853, 15.7035642157176, 16.4510219628328, 16.7474993878412, 17.3397025587698, 17.1062522298643, 16.1786066072489, 16.1786066072489, 13.7402485937614, 11.9531909618517, 10.9073964111009])
        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues, rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues, rtol=self.rtol_fitted)