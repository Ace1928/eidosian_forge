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
class TestRegularized:

    def test_regularized(self):
        import os
        from .results import glmnet_r_results
        for dtype in ('binomial', 'poisson'):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            data = np.loadtxt(os.path.join(cur_dir, 'results', 'enet_%s.csv' % dtype), delimiter=',')
            endog = data[:, 0]
            exog = data[:, 1:]
            fam = {'binomial': sm.families.Binomial, 'poisson': sm.families.Poisson}[dtype]
            for j in range(9):
                vn = 'rslt_%s_%d' % (dtype, j)
                r_result = getattr(glmnet_r_results, vn)
                L1_wt = r_result[0]
                alpha = r_result[1]
                params = r_result[2:]
                model = GLM(endog, exog, family=fam())
                sm_result = model.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                assert_allclose(params, sm_result.params, atol=0.01, rtol=0.3)

                def plf(params):
                    llf = model.loglike(params) / len(endog)
                    llf = llf - alpha * ((1 - L1_wt) * np.sum(params ** 2) / 2 + L1_wt * np.sum(np.abs(params)))
                    return llf
                llf_r = plf(params)
                llf_sm = plf(sm_result.params)
                assert_equal(np.sign(llf_sm - llf_r), 1)