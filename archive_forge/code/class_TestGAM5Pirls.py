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
class TestGAM5Pirls(CheckGAMMixin):
    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        x = data_mcycle['times'].values
        endog = data_mcycle['accel']
        cc = CyclicCubicSplines(x, df=[6], constraints='center')
        gam_cc = GLMGam(endog, smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit()
        cls.res2 = results_pls.pls5
        cls.rtol_fitted = 1e-12
        cls.covp_corrfact = 1