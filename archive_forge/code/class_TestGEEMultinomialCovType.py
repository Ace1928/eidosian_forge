from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
class TestGEEMultinomialCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):
        endog, exog, groups = load_data('gee_nominal_1.csv', icept=False)
        va = cov_struct.Independence()
        cls.mod = gee.NominalGEE(endog, exog, groups, cov_struct=va)
        cls.start_params = np.array([0.44944752, 0.45569985, -0.92007064, -0.46766728])

    def test_wrapper(self):
        endog, exog, groups = load_data('gee_nominal_1.csv', icept=False)
        endog = pd.Series(endog, name='yendog')
        exog = pd.DataFrame(exog)
        groups = pd.Series(groups, name='the_group')
        va = cov_struct.Independence()
        mod = gee.NominalGEE(endog, exog, groups, cov_struct=va)
        rslt2 = mod.fit()
        check_wrapper(rslt2)