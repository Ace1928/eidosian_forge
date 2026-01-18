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
class CheckGAMMixin:

    @classmethod
    def _init(cls):
        cc_h = CyclicCubicSplines(np.asarray(data_mcycle['times']), df=[6])
        constraints = np.atleast_2d(cc_h.basis.mean(0))
        transf = transf_constraints(constraints)
        exog = cc_h.basis.dot(transf)
        penalty_matrix = transf.T.dot(cc_h.penalty_matrices[0]).dot(transf)
        restriction = matrix_sqrt(penalty_matrix)
        return (exog, penalty_matrix, restriction)

    def test_params(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params, res2.params, rtol=1e-05)
        assert_allclose(np.asarray(res1.cov_params()), res2.Vp * self.covp_corrfact, rtol=1e-06, atol=1e-09)
        assert_allclose(res1.scale, res2.scale * self.covp_corrfact, rtol=1e-08)
        assert_allclose(np.asarray(res1.bse), res2.se * np.sqrt(self.covp_corrfact), rtol=1e-06, atol=1e-09)

    def test_fitted(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.fittedvalues, res2.fitted_values, rtol=self.rtol_fitted)

    @pytest.mark.smoke
    def test_null_smoke(self):
        self.res1.llnull