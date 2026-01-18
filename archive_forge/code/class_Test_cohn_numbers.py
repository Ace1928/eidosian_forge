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
class Test_cohn_numbers:

    def setup_method(self):
        self.df = load_basic_data()
        self.final_cols = ['lower_dl', 'upper_dl', 'nuncen_above', 'nobs_below', 'ncen_equal', 'prob_exceedance']
        self.expected_baseline = pandas.DataFrame([{'lower_dl': 2.0, 'ncen_equal': 0.0, 'nobs_below': 0.0, 'nuncen_above': 3.0, 'prob_exceedance': 1.0, 'upper_dl': 5.0}, {'lower_dl': 5.0, 'ncen_equal': 2.0, 'nobs_below': 5.0, 'nuncen_above': 0.0, 'prob_exceedance': 0.7775743707093822, 'upper_dl': 5.5}, {'lower_dl': 5.5, 'ncen_equal': 1.0, 'nobs_below': 6.0, 'nuncen_above': 2.0, 'prob_exceedance': 0.7775743707093822, 'upper_dl': 5.75}, {'lower_dl': 5.75, 'ncen_equal': 1.0, 'nobs_below': 9.0, 'nuncen_above': 10.0, 'prob_exceedance': 0.7034324942791762, 'upper_dl': 9.5}, {'lower_dl': 9.5, 'ncen_equal': 2.0, 'nobs_below': 21.0, 'nuncen_above': 2.0, 'prob_exceedance': 0.3739130434782609, 'upper_dl': 11.0}, {'lower_dl': 11.0, 'ncen_equal': 1.0, 'nobs_below': 24.0, 'nuncen_above': 11.0, 'prob_exceedance': 0.3142857142857143, 'upper_dl': numpy.inf}, {'lower_dl': numpy.nan, 'ncen_equal': numpy.nan, 'nobs_below': numpy.nan, 'nuncen_above': numpy.nan, 'prob_exceedance': 0.0, 'upper_dl': numpy.nan}])[self.final_cols]

    def test_baseline(self):
        result = ros.cohn_numbers(self.df, observations='conc', censorship='censored')
        assert_frame_equal(result, self.expected_baseline)

    def test_no_NDs(self):
        _df = self.df.copy()
        _df['qual'] = False
        result = ros.cohn_numbers(_df, observations='conc', censorship='qual')
        assert result.shape == (0, 6)