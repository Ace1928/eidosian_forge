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
class TestProbitNewton(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = Probit(data.endog, data.exog).fit(method='newton', disp=0)
        res2 = Spector.probit
        cls.res2 = res2

    def test_init_kwargs(self):
        endog = self.res1.model.endog
        exog = self.res1.model.exog
        z = np.ones(len(endog))
        with pytest.warns(ValueWarning, match='unknown kwargs'):
            Probit(endog, exog, weights=z)