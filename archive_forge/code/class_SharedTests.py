from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class SharedTests:
    index_cls: type[DatetimeIndex | PeriodIndex | TimedeltaIndex]

    @pytest.fixture
    def arr1d(self):
        """Fixture returning DatetimeArray with daily frequency."""
        data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, freq='D')
        else:
            arr = self.index_cls(data, freq='D')._data
        return arr

    def test_compare_len1_raises(self, arr1d):
        arr = arr1d
        idx = self.index_cls(arr)
        with pytest.raises(ValueError, match='Lengths must match'):
            arr == arr[:1]
        with pytest.raises(ValueError, match='Lengths must match'):
            idx <= idx[[0]]

    @pytest.mark.parametrize('result', [pd.date_range('2020', periods=3), pd.date_range('2020', periods=3, tz='UTC'), pd.timedelta_range('0 days', periods=3), pd.period_range('2020Q1', periods=3, freq='Q')])
    def test_compare_with_Categorical(self, result):
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('as_index', [True, False])
    def test_compare_categorical_dtype(self, arr1d, as_index, reverse, ordered):
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)
        left, right = (arr1d, other)
        if reverse:
            left, right = (right, left)
        ones = np.ones(arr1d.shape, dtype=bool)
        zeros = ~ones
        result = left == right
        tm.assert_numpy_array_equal(result, ones)
        result = left != right
        tm.assert_numpy_array_equal(result, zeros)
        if not reverse and (not as_index):
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)
            result = left <= right
            tm.assert_numpy_array_equal(result, ones)
            result = left > right
            tm.assert_numpy_array_equal(result, zeros)
            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self):
        data = np.arange(100, dtype='i8') * 24 * 3600 * 10 ** 9
        np.random.default_rng(2).shuffle(data)
        if self.array_cls is PeriodArray:
            arr = PeriodArray(data, dtype='period[D]')
        else:
            arr = self.index_cls(data)._data
        idx = self.index_cls._simple_new(arr)
        takers = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)
        tm.assert_index_equal(self.index_cls(result), expected)
        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)
        tm.assert_index_equal(self.index_cls(result), expected)

    @pytest.mark.parametrize('fill_value', [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    def test_take_fill_raises(self, fill_value, arr1d):
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)

    def test_take_fill(self, arr1d):
        arr = arr1d
        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT
        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT
        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    def test_take_fill_str(self, arr1d):
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        tm.assert_equal(result, expected)
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([-1, 1], allow_fill=True, fill_value='foo')

    def test_concat_same_type(self, arr1d):
        arr = arr1d
        idx = self.index_cls(arr)
        idx = idx.insert(0, NaT)
        arr = arr1d
        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        arr2 = arr.astype(object)
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))
        tm.assert_index_equal(self.index_cls(result), expected)

    def test_unbox_scalar(self, arr1d):
        result = arr1d._unbox_scalar(arr1d[0])
        expected = arr1d._ndarray.dtype.type
        assert isinstance(result, expected)
        result = arr1d._unbox_scalar(NaT)
        assert isinstance(result, expected)
        msg = f"'value' should be a {self.scalar_type.__name__}."
        with pytest.raises(ValueError, match=msg):
            arr1d._unbox_scalar('foo')

    def test_check_compatible_with(self, arr1d):
        arr1d._check_compatible_with(arr1d[0])
        arr1d._check_compatible_with(arr1d[:1])
        arr1d._check_compatible_with(NaT)

    def test_scalar_from_string(self, arr1d):
        result = arr1d._scalar_from_string(str(arr1d[0]))
        assert result == arr1d[0]

    def test_reduce_invalid(self, arr1d):
        msg = "does not support reduction 'not a method'"
        with pytest.raises(TypeError, match=msg):
            arr1d._reduce('not a method')

    @pytest.mark.parametrize('method', ['pad', 'backfill'])
    def test_fillna_method_doesnt_change_orig(self, method):
        data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.array_cls._from_sequence(data)
        arr[4] = NaT
        fill_value = arr[3] if method == 'pad' else arr[5]
        result = arr._pad_or_backfill(method=method)
        assert result[4] == fill_value
        assert arr[4] is NaT

    def test_searchsorted(self):
        data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.array_cls._from_sequence(data)
        result = arr.searchsorted(arr[1])
        assert result == 1
        result = arr.searchsorted(arr[2], side='right')
        assert result == 3
        result = arr.searchsorted(arr[1:3])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = arr.searchsorted(arr[1:3], side='right')
        expected = np.array([2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = arr.searchsorted(NaT)
        assert result == 10

    @pytest.mark.parametrize('box', [None, 'index', 'series'])
    def test_searchsorted_castable_strings(self, arr1d, box, string_storage):
        arr = arr1d
        if box is None:
            pass
        elif box == 'index':
            arr = self.index_cls(arr)
        else:
            arr = pd.Series(arr)
        result = arr.searchsorted(str(arr[1]))
        assert result == 1
        result = arr.searchsorted(str(arr[2]), side='right')
        assert result == 3
        result = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        with pytest.raises(TypeError, match=re.escape(f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got 'str' instead.")):
            arr.searchsorted('foo')
        with pd.option_context('string_storage', string_storage):
            with pytest.raises(TypeError, match=re.escape(f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got string array instead.")):
                arr.searchsorted([str(arr[1]), 'baz'])

    def test_getitem_near_implementation_bounds(self):
        i8vals = np.asarray([NaT._value + n for n in range(1, 5)], dtype='i8')
        if self.array_cls is PeriodArray:
            arr = self.array_cls(i8vals, dtype='period[ns]')
        else:
            arr = self.index_cls(i8vals, freq='ns')._data
        arr[0]
        index = pd.Index(arr)
        index[0]
        ser = pd.Series(arr)
        ser[0]

    def test_getitem_2d(self, arr1d):
        expected = type(arr1d)._simple_new(arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype)
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)
        arr2d = expected
        expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)
        result = arr2d[-1, 0]
        expected = arr1d[-1]
        assert result == expected

    def test_iter_2d(self, arr1d):
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    def test_repr_2d(self, arr1d):
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        result = repr(arr2d)
        if isinstance(arr2d, TimedeltaArray):
            expected = f"<{type(arr2d).__name__}>\n[\n['{arr1d[0]._repr_base()}'],\n['{arr1d[1]._repr_base()}'],\n['{arr1d[2]._repr_base()}']\n]\nShape: (3, 1), dtype: {arr1d.dtype}"
        else:
            expected = f"<{type(arr2d).__name__}>\n[\n['{arr1d[0]}'],\n['{arr1d[1]}'],\n['{arr1d[2]}']\n]\nShape: (3, 1), dtype: {arr1d.dtype}"
        assert result == expected

    def test_setitem(self):
        data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.index_cls(data, freq='D')._data
        arr[0] = arr[1]
        expected = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        expected[0] = expected[1]
        tm.assert_numpy_array_equal(arr.asi8, expected)
        arr[:2] = arr[-2:]
        expected[:2] = expected[-2:]
        tm.assert_numpy_array_equal(arr.asi8, expected)

    @pytest.mark.parametrize('box', [pd.Index, pd.Series, np.array, list, NumpyExtensionArray])
    def test_setitem_object_dtype(self, box, arr1d):
        expected = arr1d.copy()[::-1]
        if expected.dtype.kind in ['m', 'M']:
            expected = expected._with_freq(None)
        vals = expected
        if box is list:
            vals = list(vals)
        elif box is np.array:
            vals = np.array(vals.astype(object))
        elif box is NumpyExtensionArray:
            vals = box(np.asarray(vals, dtype=object))
        else:
            vals = box(vals).astype(object)
        arr1d[:] = vals
        tm.assert_equal(arr1d, expected)

    def test_setitem_strs(self, arr1d):
        expected = arr1d.copy()
        expected[[0, 1]] = arr1d[-2:]
        result = arr1d.copy()
        result[:2] = [str(x) for x in arr1d[-2:]]
        tm.assert_equal(result, expected)
        expected = arr1d.copy()
        expected[0] = arr1d[-1]
        result = arr1d.copy()
        result[0] = str(arr1d[-1])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('as_index', [True, False])
    def test_setitem_categorical(self, arr1d, as_index):
        expected = arr1d.copy()[::-1]
        if not isinstance(expected, PeriodArray):
            expected = expected._with_freq(None)
        cat = pd.Categorical(arr1d)
        if as_index:
            cat = pd.CategoricalIndex(cat)
        arr1d[:] = cat[::-1]
        tm.assert_equal(arr1d, expected)

    def test_setitem_raises(self, arr1d):
        arr = arr1d[:10]
        val = arr[0]
        with pytest.raises(IndexError, match='index 12 is out of bounds'):
            arr[12] = val
        with pytest.raises(TypeError, match="value should be a.* 'object'"):
            arr[0] = object()
        msg = 'cannot set using a list-like indexer with a different length'
        with pytest.raises(ValueError, match=msg):
            arr[[]] = [arr[1]]
        msg = 'cannot set using a slice indexer with a different length than'
        with pytest.raises(ValueError, match=msg):
            arr[1:1] = arr[:3]

    @pytest.mark.parametrize('box', [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(self, arr1d, box):
        msg = f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0, 1])
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0.0, 1.0])

    def test_inplace_arithmetic(self):
        data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.index_cls(data, freq='D')._data
        expected = arr + pd.Timedelta(days=1)
        arr += pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)
        expected = arr - pd.Timedelta(days=1)
        arr -= pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

    def test_shift_fill_int_deprecated(self, arr1d):
        with pytest.raises(TypeError, match='value should be a'):
            arr1d.shift(1, fill_value=1)

    def test_median(self, arr1d):
        arr = arr1d
        if len(arr) % 2 == 0:
            arr = arr[:-1]
        expected = arr[len(arr) // 2]
        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected
        arr[len(arr) // 2] = NaT
        if not isinstance(expected, Period):
            expected = arr[len(arr) // 2 - 1:len(arr) // 2 + 2].mean()
        assert arr.median(skipna=False) is NaT
        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected
        assert arr[:0].median() is NaT
        assert arr[:0].median(skipna=False) is NaT
        arr2 = arr.reshape(-1, 1)
        result = arr2.median(axis=None)
        assert type(result) is type(expected)
        assert result == expected
        assert arr2.median(axis=None, skipna=False) is NaT
        result = arr2.median(axis=0)
        expected2 = type(arr)._from_sequence([expected], dtype=arr.dtype)
        tm.assert_equal(result, expected2)
        result = arr2.median(axis=0, skipna=False)
        expected2 = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected2)
        result = arr2.median(axis=1)
        tm.assert_equal(result, arr)
        result = arr2.median(axis=1, skipna=False)
        tm.assert_equal(result, arr)

    def test_from_integer_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        data = pd.array(arr, dtype='Int64')
        if self.array_cls is PeriodArray:
            expected = self.array_cls(arr, dtype=self.example_dtype)
            result = self.array_cls(data, dtype=self.example_dtype)
        else:
            expected = self.array_cls._from_sequence(arr, dtype=self.example_dtype)
            result = self.array_cls._from_sequence(data, dtype=self.example_dtype)
        tm.assert_extension_array_equal(result, expected)