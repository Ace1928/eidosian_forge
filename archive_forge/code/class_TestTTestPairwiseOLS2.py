from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
class TestTTestPairwiseOLS2(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = ols('np.log(Days+1) ~ C(Weight) + C(Duration)', cls.data)
        cls.res = mod.fit()
        cls.term_name = 'C(Weight)'
        cls.constraints = ['C(Weight)[T.2]', 'C(Weight)[T.3]', 'C(Weight)[T.3] - C(Weight)[T.2]']