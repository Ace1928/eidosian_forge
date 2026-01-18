import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
class TestConvergence:

    @classmethod
    def setup_class(cls):
        """
        Test Binomial family with canonical logit link using star98 dataset.
        """
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.model = GLM(data.endog, data.exog, family=sm.families.Binomial())

    def _when_converged(self, atol=1e-08, rtol=0, tol_criterion='deviance'):
        for i, dev in enumerate(self.res.fit_history[tol_criterion]):
            orig = self.res.fit_history[tol_criterion][i]
            new = self.res.fit_history[tol_criterion][i + 1]
            if np.allclose(orig, new, atol=atol, rtol=rtol):
                return i
        raise ValueError("CONVERGENCE CHECK: It seems this doens't converge!")

    def test_convergence_atol_only(self):
        atol = 1e-08
        rtol = 0
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)

    def test_convergence_rtol_only(self):
        atol = 0
        rtol = 1e-08
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)

    def test_convergence_atol_rtol(self):
        atol = 1e-08
        rtol = 1e-08
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)

    def test_convergence_atol_only_params(self):
        atol = 1e-08
        rtol = 0
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol, tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)

    def test_convergence_rtol_only_params(self):
        atol = 0
        rtol = 1e-08
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol, tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)

    def test_convergence_atol_rtol_params(self):
        atol = 1e-08
        rtol = 1e-08
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol, tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)