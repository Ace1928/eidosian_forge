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
class Test__ros_plot_pos:

    def setup_method(self):
        self.cohn = load_basic_cohn()

    def test_uncensored_1(self):
        row = {'censored': False, 'det_limit_index': 2, 'rank': 1}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 0.24713958810068648)

    def test_uncensored_2(self):
        row = {'censored': False, 'det_limit_index': 2, 'rank': 12}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 0.5189931350114417)

    def test_censored_1(self):
        row = {'censored': True, 'det_limit_index': 5, 'rank': 4}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 1.3714285714285714)

    def test_censored_2(self):
        row = {'censored': True, 'det_limit_index': 4, 'rank': 2}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 0.41739130434782606)