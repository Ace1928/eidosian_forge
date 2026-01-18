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
class TestPoissonL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 10
        cls.m = 7
        rand_data = load_randhie()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog = sm.add_constant(rand_exog, prepend=True)
        exog_no_PSI = rand_exog[:, :cls.m]
        mod_unreg = sm.Poisson(rand_data.endog, exog_no_PSI)
        cls.res_unreg = mod_unreg.fit(method='newton', disp=False)
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars)
        alpha[:cls.m] = 0
        cls.res_reg = sm.Poisson(rand_data.endog, rand_exog).fit_regularized(method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000, trim_mode='auto')