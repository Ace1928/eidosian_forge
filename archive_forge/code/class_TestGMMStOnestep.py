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
class TestGMMStOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [0.0005, 5e-05]
        cls.bse_tol = [0.007, 0.0005]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=0, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0})
        cls.res1 = res
        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    def test_bse_other(self):
        res1, res2 = (self.res1, self.res2)
        bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False)))
        q = self.res1.model.gmmobjective(self.res1.params, np.linalg.inv(self.res1.weights))

    @pytest.mark.xfail(reason='q vs Q comparison fails', raises=AssertionError, strict=True)
    def test_other(self):
        super().test_other()