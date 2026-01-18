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
class TestMNLogitLBFGSBaseZero(CheckMNLogitBaseZero):

    @classmethod
    def setup_class(cls):
        cls.data = data = load_anes96()
        exog = data.exog
        exog = sm.add_constant(exog, prepend=False)
        mymodel = MNLogit(data.endog, exog)
        cls.res1 = mymodel.fit(method='lbfgs', disp=0, maxiter=50000, m=40, pgtol=1e-10, factr=5.0, loglike_and_score=mymodel.loglike_and_score)
        res2 = Anes.mnlogit_basezero
        cls.res2 = res2