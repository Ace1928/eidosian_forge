from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
class TestAddTrend:

    @classmethod
    def setup_class(cls):
        cls.n = 200
        cls.arr_1d = np.arange(float(cls.n))
        cls.arr_2d = np.tile(np.arange(float(cls.n))[:, None], 2)
        cls.c = np.ones(cls.n)
        cls.t = np.arange(1.0, cls.n + 1)

    def test_series(self):
        s = pd.Series(self.arr_1d)
        appended = tools.add_trend(s)
        expected = pd.DataFrame(s)
        expected['const'] = self.c
        assert_frame_equal(expected, appended)
        prepended = tools.add_trend(s, prepend=True)
        expected = pd.DataFrame(s)
        expected.insert(0, 'const', self.c)
        assert_frame_equal(expected, prepended)
        s = pd.Series(self.arr_1d)
        appended = tools.add_trend(s, trend='ct')
        expected = pd.DataFrame(s)
        expected['const'] = self.c
        expected['trend'] = self.t
        assert_frame_equal(expected, appended)

    def test_dataframe(self):
        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df)
        expected = df.copy()
        expected['const'] = self.c
        assert_frame_equal(expected, appended)
        prepended = tools.add_trend(df, prepend=True)
        expected = df.copy()
        expected.insert(0, 'const', self.c)
        assert_frame_equal(expected, prepended)
        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df, trend='t')
        expected = df.copy()
        expected['trend'] = self.t
        assert_frame_equal(expected, appended)
        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df, trend='ctt')
        expected = df.copy()
        expected['const'] = self.c
        expected['trend'] = self.t
        expected['trend_squared'] = self.t ** 2
        assert_frame_equal(expected, appended)

    def test_duplicate_const(self):
        assert_raises(ValueError, tools.add_trend, x=self.c, trend='c', has_constant='raise')
        assert_raises(ValueError, tools.add_trend, x=self.c, trend='ct', has_constant='raise')
        df = pd.DataFrame(self.c)
        assert_raises(ValueError, tools.add_trend, x=df, trend='c', has_constant='raise')
        assert_raises(ValueError, tools.add_trend, x=df, trend='ct', has_constant='raise')
        skipped = tools.add_trend(self.c, trend='c')
        assert_equal(skipped, self.c[:, None])
        skipped_const = tools.add_trend(self.c, trend='ct', has_constant='skip')
        expected = np.vstack((self.c, self.t)).T
        assert_equal(skipped_const, expected)
        added = tools.add_trend(self.c, trend='c', has_constant='add')
        expected = np.vstack((self.c, self.c)).T
        assert_equal(added, expected)
        added = tools.add_trend(self.c, trend='ct', has_constant='add')
        expected = np.vstack((self.c, self.c, self.t)).T
        assert_equal(added, expected)

    def test_dataframe_duplicate(self):
        df = pd.DataFrame(self.arr_2d, columns=['const', 'trend'])
        tools.add_trend(df, trend='ct')
        tools.add_trend(df, trend='ct', prepend=True)

    def test_array(self):
        base = np.vstack((self.arr_1d, self.c, self.t, self.t ** 2)).T
        assert_equal(tools.add_trend(self.arr_1d), base[:, :2])
        assert_equal(tools.add_trend(self.arr_1d, trend='t'), base[:, [0, 2]])
        assert_equal(tools.add_trend(self.arr_1d, trend='ct'), base[:, :3])
        assert_equal(tools.add_trend(self.arr_1d, trend='ctt'), base)
        base = np.hstack((self.c[:, None], self.t[:, None], self.t[:, None] ** 2, self.arr_2d))
        assert_equal(tools.add_trend(self.arr_2d, prepend=True), base[:, [0, 3, 4]])
        assert_equal(tools.add_trend(self.arr_2d, trend='t', prepend=True), base[:, [1, 3, 4]])
        assert_equal(tools.add_trend(self.arr_2d, trend='ct', prepend=True), base[:, [0, 1, 3, 4]])
        assert_equal(tools.add_trend(self.arr_2d, trend='ctt', prepend=True), base)

    def test_unknown_trend(self):
        assert_raises(ValueError, tools.add_trend, x=self.arr_1d, trend='unknown')

    def test_trend_n(self):
        assert_equal(tools.add_trend(self.arr_1d, 'n'), self.arr_1d)
        assert tools.add_trend(self.arr_1d, 'n') is not self.arr_1d
        assert_equal(tools.add_trend(self.arr_2d, 'n'), self.arr_2d)
        assert tools.add_trend(self.arr_2d, 'n') is not self.arr_2d