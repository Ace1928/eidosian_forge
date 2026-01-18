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
class TestGMMSt1(CheckGMM):

    @classmethod
    def setup_class(cls):
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(start, maxiter=10, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0}, wargs={'centered': False})
        cls.res1 = res10
        from .results_gmm_griliches_iter import results
        cls.res2 = results