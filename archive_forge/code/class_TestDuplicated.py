from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
class TestDuplicated:

    def test_duplicated_with_nas(self):
        keys = np.array([0, 1, np.nan, 0, 2, np.nan], dtype=object)
        result = algos.duplicated(keys)
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.duplicated(keys, keep='first')
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.duplicated(keys, keep='last')
        expected = np.array([True, False, True, False, False, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.duplicated(keys, keep=False)
        expected = np.array([True, False, True, True, False, True])
        tm.assert_numpy_array_equal(result, expected)
        keys = np.empty(8, dtype=object)
        for i, t in enumerate(zip([0, 0, np.nan, np.nan] * 2, [0, np.nan, 0, np.nan] * 2)):
            keys[i] = t
        result = algos.duplicated(keys)
        falses = [False] * 4
        trues = [True] * 4
        expected = np.array(falses + trues)
        tm.assert_numpy_array_equal(result, expected)
        result = algos.duplicated(keys, keep='last')
        expected = np.array(trues + falses)
        tm.assert_numpy_array_equal(result, expected)
        result = algos.duplicated(keys, keep=False)
        expected = np.array(trues + trues)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('case', [np.array([1, 2, 1, 5, 3, 2, 4, 1, 5, 6]), np.array([1.1, 2.2, 1.1, np.nan, 3.3, 2.2, 4.4, 1.1, np.nan, 6.6]), np.array([1 + 1j, 2 + 2j, 1 + 1j, 5 + 5j, 3 + 3j, 2 + 2j, 4 + 4j, 1 + 1j, 5 + 5j, 6 + 6j]), np.array(['a', 'b', 'a', 'e', 'c', 'b', 'd', 'a', 'e', 'f'], dtype=object), np.array([1, 2 ** 63, 1, 3 ** 5, 10, 2 ** 63, 39, 1, 3 ** 5, 7], dtype=np.uint64)])
    def test_numeric_object_likes(self, case):
        exp_first = np.array([False, False, True, False, False, True, False, True, True, False])
        exp_last = np.array([True, True, True, True, False, False, False, False, False, False])
        exp_false = exp_first | exp_last
        res_first = algos.duplicated(case, keep='first')
        tm.assert_numpy_array_equal(res_first, exp_first)
        res_last = algos.duplicated(case, keep='last')
        tm.assert_numpy_array_equal(res_last, exp_last)
        res_false = algos.duplicated(case, keep=False)
        tm.assert_numpy_array_equal(res_false, exp_false)
        for idx in [Index(case), Index(case, dtype='category')]:
            res_first = idx.duplicated(keep='first')
            tm.assert_numpy_array_equal(res_first, exp_first)
            res_last = idx.duplicated(keep='last')
            tm.assert_numpy_array_equal(res_last, exp_last)
            res_false = idx.duplicated(keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)
        for s in [Series(case), Series(case, dtype='category')]:
            res_first = s.duplicated(keep='first')
            tm.assert_series_equal(res_first, Series(exp_first))
            res_last = s.duplicated(keep='last')
            tm.assert_series_equal(res_last, Series(exp_last))
            res_false = s.duplicated(keep=False)
            tm.assert_series_equal(res_false, Series(exp_false))

    def test_datetime_likes(self):
        dt = ['2011-01-01', '2011-01-02', '2011-01-01', 'NaT', '2011-01-03', '2011-01-02', '2011-01-04', '2011-01-01', 'NaT', '2011-01-06']
        td = ['1 days', '2 days', '1 days', 'NaT', '3 days', '2 days', '4 days', '1 days', 'NaT', '6 days']
        cases = [np.array([Timestamp(d) for d in dt]), np.array([Timestamp(d, tz='US/Eastern') for d in dt]), np.array([Period(d, freq='D') for d in dt]), np.array([np.datetime64(d) for d in dt]), np.array([Timedelta(d) for d in td])]
        exp_first = np.array([False, False, True, False, False, True, False, True, True, False])
        exp_last = np.array([True, True, True, True, False, False, False, False, False, False])
        exp_false = exp_first | exp_last
        for case in cases:
            res_first = algos.duplicated(case, keep='first')
            tm.assert_numpy_array_equal(res_first, exp_first)
            res_last = algos.duplicated(case, keep='last')
            tm.assert_numpy_array_equal(res_last, exp_last)
            res_false = algos.duplicated(case, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)
            for idx in [Index(case), Index(case, dtype='category'), Index(case, dtype=object)]:
                res_first = idx.duplicated(keep='first')
                tm.assert_numpy_array_equal(res_first, exp_first)
                res_last = idx.duplicated(keep='last')
                tm.assert_numpy_array_equal(res_last, exp_last)
                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)
            for s in [Series(case), Series(case, dtype='category'), Series(case, dtype=object)]:
                res_first = s.duplicated(keep='first')
                tm.assert_series_equal(res_first, Series(exp_first))
                res_last = s.duplicated(keep='last')
                tm.assert_series_equal(res_last, Series(exp_last))
                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    @pytest.mark.parametrize('case', [Index([1, 2, 3]), pd.RangeIndex(0, 3)])
    def test_unique_index(self, case):
        assert case.is_unique is True
        tm.assert_numpy_array_equal(case.duplicated(), np.array([False, False, False]))

    @pytest.mark.parametrize('arr, uniques', [([(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0), (1, 1)]), ([('b', 'c'), ('a', 'b'), ('a', 'b'), ('b', 'c')], [('b', 'c'), ('a', 'b')]), ([('a', 1), ('b', 2), ('a', 3), ('a', 1)], [('a', 1), ('b', 2), ('a', 3)])])
    def test_unique_tuples(self, arr, uniques):
        expected = np.empty(len(uniques), dtype=object)
        expected[:] = uniques
        msg = 'unique with argument that is not not a Series'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.unique(arr)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('array,expected', [([1 + 1j, 0, 1, 1j, 1 + 2j, 1 + 2j], np.array([1 + 1j, 0j, 1 + 0j, 1j, 1 + 2j], dtype=object))])
    def test_unique_complex_numbers(self, array, expected):
        msg = 'unique with argument that is not not a Series'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.unique(array)
        tm.assert_numpy_array_equal(result, expected)