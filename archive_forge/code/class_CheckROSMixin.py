from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
class CheckROSMixin:

    def test_ros_df(self):
        result = ros.impute_ros(self.rescol, self.cencol, df=self.df)
        npt.assert_array_almost_equal(sorted(result), sorted(self.expected_final), decimal=self.decimal)

    def test_ros_arrays(self):
        result = ros.impute_ros(self.df[self.rescol], self.df[self.cencol], df=None)
        npt.assert_array_almost_equal(sorted(result), sorted(self.expected_final), decimal=self.decimal)

    def test_cohn(self):
        cols = ['nuncen_above', 'nobs_below', 'ncen_equal', 'prob_exceedance']
        cohn = ros.cohn_numbers(self.df, self.rescol, self.cencol)
        assert_frame_equal(np.round(cohn[cols], 3), np.round(self.expected_cohn[cols], 3))