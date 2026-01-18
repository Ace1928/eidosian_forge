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
class TestMNLogitNewtonBaseZero(CheckMNLogitBaseZero):

    @classmethod
    def setup_class(cls):
        cls.data = data = load_anes96()
        exog = data.exog
        exog = sm.add_constant(exog, prepend=False)
        cls.res1 = MNLogit(data.endog, exog).fit(method='newton', disp=0)
        res2 = Anes.mnlogit_basezero
        cls.res2 = res2