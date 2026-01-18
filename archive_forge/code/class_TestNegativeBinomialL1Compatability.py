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
class TestNegativeBinomialL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 10
        cls.m = 7
        rand_data = load_randhie()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog_st = (rand_exog - rand_exog.mean(0)) / rand_exog.std(0)
        rand_exog = sm.add_constant(rand_exog_st, prepend=True)
        exog_no_PSI = rand_exog[:, :cls.m]
        mod_unreg = sm.NegativeBinomial(rand_data.endog, exog_no_PSI)
        cls.res_unreg = mod_unreg.fit(method='newton', disp=False)
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars + 1)
        alpha[:cls.m] = 0
        alpha[-1] = 0
        mod_reg = sm.NegativeBinomial(rand_data.endog, rand_exog)
        cls.res_reg = mod_reg.fit_regularized(method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000, trim_mode='auto')
        cls.k_extra = 1