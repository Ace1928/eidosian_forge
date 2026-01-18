from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
class TestGMMStOneiterNO_Linear(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [5e-09, 1e-09]
        cls.bse_tol = [5e-10, 1e-10]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.LinearIVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-08, 'disp': 0}, wargs={'centered': False}, has_optimal_weights=False)
        cls.res1 = res
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0}, wargs={'centered': False}, has_optimal_weights=False)
        cls.res3 = res
        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason='q vs Q comparison fails', raises=AssertionError, strict=True)
    def test_other(self):
        super().test_other()