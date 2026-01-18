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
class TestL1AlphaZeroProbit(CompareL11D):

    @classmethod
    def setup_class(cls):
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.res1 = Probit(data.endog, data.exog).fit_regularized(method='l1', alpha=0, disp=0, acc=1e-15, maxiter=1000, trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = Probit(data.endog, data.exog).fit(disp=0, tol=1e-15)