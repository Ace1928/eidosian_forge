import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
class TestSummaryLabels:
    """
    Test that the labels are correctly set in the summary table"""

    @classmethod
    def setup_class(cls):
        y = [1, 1, 4, 2] * 4
        x = add_constant([1, 2, 3, 4] * 4)
        cls.mod = OLS(endog=y, exog=x).fit()

    def test_summary_col_r2(self):
        table = summary_col(results=self.mod, include_r2=True)
        assert 'R-squared  ' in str(table)
        assert 'R-squared Adj.' in str(table)

    def test_absence_of_r2(self):
        table = summary_col(results=self.mod, include_r2=False)
        assert 'R-squared' not in str(table)
        assert 'R-squared Adj.' not in str(table)