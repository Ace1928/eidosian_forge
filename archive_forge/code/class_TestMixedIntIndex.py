from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
class TestMixedIntIndex:

    @pytest.fixture
    def simple_index(self) -> Index:
        return Index([0, 'a', 1, 'b', 2, 'c'])

    def test_argsort(self, simple_index):
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            index.argsort()

    def test_numpy_argsort(self, simple_index):
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            np.argsort(index)

    def test_copy_name(self, simple_index):
        index = simple_index
        first = type(index)(index, copy=True, name='mario')
        second = type(first)(first, copy=False)
        assert first is not second
        tm.assert_index_equal(first, second)
        assert first.name == 'mario'
        assert second.name == 'mario'
        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])
        s3 = s1 * s2
        assert s3.index.name == 'mario'

    def test_copy_name2(self):
        index = Index([1, 2], name='MyName')
        index1 = index.copy()
        tm.assert_index_equal(index, index1)
        index2 = index.copy(name='NewName')
        tm.assert_index_equal(index, index2, check_names=False)
        assert index.name == 'MyName'
        assert index2.name == 'NewName'

    def test_unique_na(self):
        idx = Index([2, np.nan, 2, 1], name='my_index')
        expected = Index([2, np.nan, 1], name='my_index')
        result = idx.unique()
        tm.assert_index_equal(result, expected)

    def test_logical_compat(self, simple_index):
        index = simple_index
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize('how', ['any', 'all'])
    @pytest.mark.parametrize('dtype', [None, object, 'category'])
    @pytest.mark.parametrize('vals,expected', [([1, 2, 3], [1, 2, 3]), ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]), (['A', 'B', 'C'], ['A', 'B', 'C']), (['A', np.nan, 'B', 'C'], ['A', 'B', 'C'])])
    def test_dropna(self, how, dtype, vals, expected):
        index = Index(vals, dtype=dtype)
        result = index.dropna(how=how)
        expected = Index(expected, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('how', ['any', 'all'])
    @pytest.mark.parametrize('index,expected', [(DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']), DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])), (DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', pd.NaT]), DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])), (TimedeltaIndex(['1 days', '2 days', '3 days']), TimedeltaIndex(['1 days', '2 days', '3 days'])), (TimedeltaIndex([pd.NaT, '1 days', '2 days', '3 days', pd.NaT]), TimedeltaIndex(['1 days', '2 days', '3 days'])), (PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'), PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M')), (PeriodIndex(['2012-02', '2012-04', 'NaT', '2012-05'], freq='M'), PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'))])
    def test_dropna_dt_like(self, how, index, expected):
        result = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self):
        msg = 'invalid how option: xxx'
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3]).dropna(how='xxx')

    @pytest.mark.parametrize('index', [Index([np.nan]), Index([np.nan, 1]), Index([1, 2, np.nan]), Index(['a', 'b', np.nan]), pd.to_datetime(['NaT']), pd.to_datetime(['NaT', '2000-01-01']), pd.to_datetime(['2000-01-01', 'NaT', '2000-01-02']), pd.to_timedelta(['1 day', 'NaT'])])
    def test_is_monotonic_na(self, index):
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is False

    @pytest.mark.parametrize('dtype', ['f8', 'm8[ns]', 'M8[us]'])
    @pytest.mark.parametrize('unique_first', [True, False])
    def test_is_monotonic_unique_na(self, dtype, unique_first):
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    def test_int_name_format(self, frame_or_series):
        index = Index(['a', 'b', 'c'], name=0)
        result = frame_or_series(list(range(3)), index=index)
        assert '0' in repr(result)

    def test_str_to_bytes_raises(self):
        index = Index([str(x) for x in range(10)])
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(index)

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:FutureWarning')
    def test_index_with_tuple_bool(self):
        idx = Index([('a', 'b'), ('b', 'c'), ('c', 'a')])
        result = idx == ('c', 'a')
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)