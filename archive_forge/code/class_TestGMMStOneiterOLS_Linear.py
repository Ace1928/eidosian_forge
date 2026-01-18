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
class TestGMMStOneiterOLS_Linear(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [1e-11, 1e-12]
        cls.bse_tol = [1e-12, 1e-12]
        exog = exog_st
        res_ols = OLS(endog, exog).fit()
        start = np.ones(len(res_ols.params))
        nobs, k_instr = instrument.shape
        w0inv = np.dot(exog.T, exog) / nobs
        mod = gmm.LinearIVGMM(endog, exog, exog)
        res = mod.fit(start, maxiter=0, inv_weights=w0inv, optim_args={'disp': 0}, weights_method='iid', wargs={'centered': False, 'ddof': 'k_params'}, has_optimal_weights=True)
        res.use_t = True
        res.df_resid = res.nobs - len(res.params)
        cls.res1 = res
        cls.res2 = res_ols

    @pytest.mark.xfail(reason='RegressionResults has no `Q` attribute', raises=AttributeError, strict=True)
    def test_other(self):
        super().test_other()