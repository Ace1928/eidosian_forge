from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class TestProbitCG(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector.probit
        cls.res2 = res2
        from statsmodels.tools.transform_model import StandardizeTransform
        transf = StandardizeTransform(data.exog)
        exog_st = transf(data.exog)
        res1_st = Probit(data.endog, exog_st).fit(method='cg', disp=0, maxiter=1000, gtol=1e-08)
        start_params = transf.transform_params(res1_st.params)
        assert_allclose(start_params, res2.params, rtol=1e-05, atol=1e-06)
        cls.res1 = Probit(data.endog, data.exog).fit(start_params=start_params, method='cg', maxiter=1000, gtol=1e-05, disp=0)
        assert_array_less(cls.res1.mle_retvals['fcalls'], 100)