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
class TestPenalizedLogitOraclePenalized(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        modp = LogitPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)
        mod = LogitPenalized(y, x, penal=cls.penalty)
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=False, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.001