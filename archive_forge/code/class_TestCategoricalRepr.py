from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestCategoricalRepr:

    def test_categorical_repr_unicode(self):

        class County:
            name = 'San Sebastián'
            state = 'PR'

            def __repr__(self) -> str:
                return self.name + ', ' + self.state
        cat = Categorical([County() for _ in range(61)])
        idx = Index(cat)
        ser = idx.to_series()
        repr(ser)
        str(ser)

    def test_categorical_repr(self):
        a = Series(Categorical([1, 2, 3, 4]))
        exp = '0    1\n1    2\n2    3\n3    4\n' + 'dtype: category\nCategories (4, int64): [1, 2, 3, 4]'
        assert exp == a.__str__()
        a = Series(Categorical(['a', 'b'] * 25))
        exp = '0     a\n1     b\n' + '     ..\n' + '48    a\n49    b\n' + "Length: 50, dtype: category\nCategories (2, object): ['a', 'b']"
        with option_context('display.max_rows', 5):
            assert exp == repr(a)
        levs = list('abcdefghijklmnopqrstuvwxyz')
        a = Series(Categorical(['a', 'b'], categories=levs, ordered=True))
        exp = '0    a\n1    b\n' + "dtype: category\nCategories (26, object): ['a' < 'b' < 'c' < 'd' ... 'w' < 'x' < 'y' < 'z']"
        assert exp == a.__str__()

    def test_categorical_series_repr(self):
        s = Series(Categorical([1, 2, 3]))
        exp = '0    1\n1    2\n2    3\ndtype: category\nCategories (3, int64): [1, 2, 3]'
        assert repr(s) == exp
        s = Series(Categorical(np.arange(10)))
        exp = f'0    0\n1    1\n2    2\n3    3\n4    4\n5    5\n6    6\n7    7\n8    8\n9    9\ndtype: category\nCategories (10, {np.int_().dtype}): [0, 1, 2, 3, ..., 6, 7, 8, 9]'
        assert repr(s) == exp

    def test_categorical_series_repr_ordered(self):
        s = Series(Categorical([1, 2, 3], ordered=True))
        exp = '0    1\n1    2\n2    3\ndtype: category\nCategories (3, int64): [1 < 2 < 3]'
        assert repr(s) == exp
        s = Series(Categorical(np.arange(10), ordered=True))
        exp = f'0    0\n1    1\n2    2\n3    3\n4    4\n5    5\n6    6\n7    7\n8    8\n9    9\ndtype: category\nCategories (10, {np.int_().dtype}): [0 < 1 < 2 < 3 ... 6 < 7 < 8 < 9]'
        assert repr(s) == exp

    def test_categorical_series_repr_datetime(self):
        idx = date_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx))
        exp = '0   2011-01-01 09:00:00\n1   2011-01-01 10:00:00\n2   2011-01-01 11:00:00\n3   2011-01-01 12:00:00\n4   2011-01-01 13:00:00\ndtype: category\nCategories (5, datetime64[ns]): [2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00,\n                                 2011-01-01 12:00:00, 2011-01-01 13:00:00]'
        assert repr(s) == exp
        idx = date_range('2011-01-01 09:00', freq='H', periods=5, tz='US/Eastern')
        s = Series(Categorical(idx))
        exp = '0   2011-01-01 09:00:00-05:00\n1   2011-01-01 10:00:00-05:00\n2   2011-01-01 11:00:00-05:00\n3   2011-01-01 12:00:00-05:00\n4   2011-01-01 13:00:00-05:00\ndtype: category\nCategories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n                                             2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n                                             2011-01-01 13:00:00-05:00]'
        assert repr(s) == exp

    def test_categorical_series_repr_datetime_ordered(self):
        idx = date_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0   2011-01-01 09:00:00\n1   2011-01-01 10:00:00\n2   2011-01-01 11:00:00\n3   2011-01-01 12:00:00\n4   2011-01-01 13:00:00\ndtype: category\nCategories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <\n                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]'
        assert repr(s) == exp
        idx = date_range('2011-01-01 09:00', freq='H', periods=5, tz='US/Eastern')
        s = Series(Categorical(idx, ordered=True))
        exp = '0   2011-01-01 09:00:00-05:00\n1   2011-01-01 10:00:00-05:00\n2   2011-01-01 11:00:00-05:00\n3   2011-01-01 12:00:00-05:00\n4   2011-01-01 13:00:00-05:00\ndtype: category\nCategories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <\n                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <\n                                             2011-01-01 13:00:00-05:00]'
        assert repr(s) == exp

    def test_categorical_series_repr_period(self):
        idx = period_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx))
        exp = '0    2011-01-01 09:00\n1    2011-01-01 10:00\n2    2011-01-01 11:00\n3    2011-01-01 12:00\n4    2011-01-01 13:00\ndtype: category\nCategories (5, period[H]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,\n                            2011-01-01 13:00]'
        assert repr(s) == exp
        idx = period_range('2011-01', freq='M', periods=5)
        s = Series(Categorical(idx))
        exp = '0    2011-01\n1    2011-02\n2    2011-03\n3    2011-04\n4    2011-05\ndtype: category\nCategories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]'
        assert repr(s) == exp

    def test_categorical_series_repr_period_ordered(self):
        idx = period_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0    2011-01-01 09:00\n1    2011-01-01 10:00\n2    2011-01-01 11:00\n3    2011-01-01 12:00\n4    2011-01-01 13:00\ndtype: category\nCategories (5, period[H]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <\n                            2011-01-01 13:00]'
        assert repr(s) == exp
        idx = period_range('2011-01', freq='M', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0    2011-01\n1    2011-02\n2    2011-03\n3    2011-04\n4    2011-05\ndtype: category\nCategories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]'
        assert repr(s) == exp

    def test_categorical_series_repr_timedelta(self):
        idx = timedelta_range('1 days', periods=5)
        s = Series(Categorical(idx))
        exp = '0   1 days\n1   2 days\n2   3 days\n3   4 days\n4   5 days\ndtype: category\nCategories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]'
        assert repr(s) == exp
        idx = timedelta_range('1 hours', periods=10)
        s = Series(Categorical(idx))
        exp = '0   0 days 01:00:00\n1   1 days 01:00:00\n2   2 days 01:00:00\n3   3 days 01:00:00\n4   4 days 01:00:00\n5   5 days 01:00:00\n6   6 days 01:00:00\n7   7 days 01:00:00\n8   8 days 01:00:00\n9   9 days 01:00:00\ndtype: category\nCategories (10, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,\n                                   3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00,\n                                   8 days 01:00:00, 9 days 01:00:00]'
        assert repr(s) == exp

    def test_categorical_series_repr_timedelta_ordered(self):
        idx = timedelta_range('1 days', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0   1 days\n1   2 days\n2   3 days\n3   4 days\n4   5 days\ndtype: category\nCategories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]'
        assert repr(s) == exp
        idx = timedelta_range('1 hours', periods=10)
        s = Series(Categorical(idx, ordered=True))
        exp = '0   0 days 01:00:00\n1   1 days 01:00:00\n2   2 days 01:00:00\n3   3 days 01:00:00\n4   4 days 01:00:00\n5   5 days 01:00:00\n6   6 days 01:00:00\n7   7 days 01:00:00\n8   8 days 01:00:00\n9   9 days 01:00:00\ndtype: category\nCategories (10, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <\n                                   3 days 01:00:00 ... 6 days 01:00:00 < 7 days 01:00:00 <\n                                   8 days 01:00:00 < 9 days 01:00:00]'
        assert repr(s) == exp