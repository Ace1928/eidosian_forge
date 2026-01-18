import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
class TestPenalizedPoissonOracle(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        modp = Poisson(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(disp=0)
        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005