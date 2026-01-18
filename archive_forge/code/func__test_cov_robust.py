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
def _test_cov_robust(self):
    res1 = self.res1
    res2 = self.res2
    pw = res1.model.pen_weight
    res1 = res1.model.fit(pen_weight=pw, cov_type='HC0')
    assert_allclose(np.asarray(res1.cov_params()), res2.Ve * self.covp_corrfact, rtol=0.0001)