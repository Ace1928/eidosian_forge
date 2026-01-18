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
class CheckPairwise:

    def test_default(self):
        res = self.res
        tt = res.t_test(self.constraints)
        pw = res.t_test_pairwise(self.term_name)
        pw_frame = pw.result_frame
        assert_allclose(pw_frame.iloc[:, :6].values, tt.summary_frame().values)