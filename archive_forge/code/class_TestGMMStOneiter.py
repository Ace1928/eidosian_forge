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
class TestGMMStOneiter(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [0.0005, 5e-05]
        cls.bse_tol = [0.007, 0.0005]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0})
        cls.res1 = res
        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason='q vs Q comparison fails', raises=AssertionError, strict=True)
    def test_other(self):
        super().test_other()

    def test_bse_other(self):
        res1, res2 = (self.res1, self.res2)
        moms = res1.model.momcond(res1.params)
        w = res1.model.calc_weightmatrix(moms)
        bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False, weights=res1.weights)))
        bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False)))
        nobs = instrument.shape[0]
        w0inv = np.dot(instrument.T, instrument) / nobs
        q = self.res1.model.gmmobjective(self.res1.params, w)