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
class TestProbitMinimizeDogleg(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector.probit
        cls.res2 = res2
        fit = Probit(data.endog, data.exog).fit
        cls.res1 = fit(method='minimize', disp=0, niter=5, tol=1e-08, min_method='dogleg')