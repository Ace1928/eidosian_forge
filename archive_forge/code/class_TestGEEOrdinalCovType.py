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
class TestGEEOrdinalCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):
        family = families.Binomial()
        endog, exog, groups = load_data('gee_ordinal_1.csv', icept=False)
        va = cov_struct.GlobalOddsRatio('ordinal')
        cls.mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
        cls.start_params = np.array([1.09250002, 0.0217443, -0.39851092, -0.01812116, 0.03023969, 1.18258516, 0.01803453, -1.10203381])

    def test_wrapper(self):
        endog, exog, groups = load_data('gee_ordinal_1.csv', icept=False)
        endog = pd.Series(endog, name='yendog')
        exog = pd.DataFrame(exog)
        groups = pd.Series(groups, name='the_group')
        family = families.Binomial()
        va = cov_struct.GlobalOddsRatio('ordinal')
        mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
        rslt2 = mod.fit()
        check_wrapper(rslt2)