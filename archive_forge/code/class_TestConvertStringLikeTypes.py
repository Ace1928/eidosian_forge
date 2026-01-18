import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
class TestConvertStringLikeTypes:

    def test_pandas_unicode(self):
        repeats = 1000
        values = ['foo', None, 'bar', 'mañana', np.nan]
        df = pd.DataFrame({'strings': values * repeats})
        field = pa.field('strings', pa.string())
        schema = pa.schema([field])
        ex_values = ['foo', None, 'bar', 'mañana', None]
        expected = pd.DataFrame({'strings': ex_values * repeats})
        _check_pandas_roundtrip(df, expected=expected, expected_schema=schema)

    def test_bytes_to_binary(self):
        values = ['qux', b'foo', None, bytearray(b'barz'), 'qux', np.nan]
        df = pd.DataFrame({'strings': values})
        table = pa.Table.from_pandas(df)
        assert table[0].type == pa.binary()
        values2 = [b'qux', b'foo', None, b'barz', b'qux', None]
        expected = pd.DataFrame({'strings': values2})
        _check_pandas_roundtrip(df, expected)

    @pytest.mark.large_memory
    def test_bytes_exceed_2gb(self):
        v1 = b'x' * 100000000
        v2 = b'x' * 147483646
        df = pd.DataFrame({'strings': [v1] * 20 + [v2] + ['x'] * 20})
        arr = pa.array(df['strings'])
        assert isinstance(arr, pa.ChunkedArray)
        assert arr.num_chunks == 2
        arr = None
        table = pa.Table.from_pandas(df)
        assert table[0].num_chunks == 2

    @pytest.mark.large_memory
    @pytest.mark.parametrize('char', ['x', b'x'])
    def test_auto_chunking_pandas_series_of_strings(self, char):
        v1 = char * 100000000
        v2 = char * 147483646
        df = pd.DataFrame({'strings': [[v1]] * 20 + [[v2]] + [[b'x']]})
        arr = pa.array(df['strings'], from_pandas=True)
        arr.validate(full=True)
        assert isinstance(arr, pa.ChunkedArray)
        assert arr.num_chunks == 2
        assert len(arr.chunk(0)) == 21
        assert len(arr.chunk(1)) == 1

    def test_fixed_size_bytes(self):
        values = [b'foo', None, bytearray(b'bar'), None, None, b'hey']
        df = pd.DataFrame({'strings': values})
        schema = pa.schema([pa.field('strings', pa.binary(3))])
        table = pa.Table.from_pandas(df, schema=schema)
        assert table.schema[0].type == schema[0].type
        assert table.schema[0].name == schema[0].name
        result = table.to_pandas()
        tm.assert_frame_equal(result, df)

    def test_fixed_size_bytes_does_not_accept_varying_lengths(self):
        values = [b'foo', None, b'ba', None, None, b'hey']
        df = pd.DataFrame({'strings': values})
        schema = pa.schema([pa.field('strings', pa.binary(3))])
        with pytest.raises(pa.ArrowInvalid):
            pa.Table.from_pandas(df, schema=schema)

    def test_variable_size_bytes(self):
        s = pd.Series([b'123', b'', b'a', None])
        _check_series_roundtrip(s, type_=pa.binary())

    def test_binary_from_bytearray(self):
        s = pd.Series([bytearray(b'123'), bytearray(b''), bytearray(b'a'), None])
        _check_series_roundtrip(s, type_=pa.binary())
        _check_series_roundtrip(s, expected_pa_type=pa.binary())

    def test_large_binary(self):
        s = pd.Series([b'123', b'', b'a', None])
        _check_series_roundtrip(s, type_=pa.large_binary())
        df = pd.DataFrame({'a': s})
        _check_pandas_roundtrip(df, schema=pa.schema([('a', pa.large_binary())]))

    def test_large_string(self):
        s = pd.Series(['123', '', 'a', None])
        _check_series_roundtrip(s, type_=pa.large_string())
        df = pd.DataFrame({'a': s})
        _check_pandas_roundtrip(df, schema=pa.schema([('a', pa.large_string())]))

    def test_table_empty_str(self):
        values = ['', '', '', '', '']
        df = pd.DataFrame({'strings': values})
        field = pa.field('strings', pa.string())
        schema = pa.schema([field])
        table = pa.Table.from_pandas(df, schema=schema)
        result1 = table.to_pandas(strings_to_categorical=False)
        expected1 = pd.DataFrame({'strings': values})
        tm.assert_frame_equal(result1, expected1, check_dtype=True)
        result2 = table.to_pandas(strings_to_categorical=True)
        expected2 = pd.DataFrame({'strings': pd.Categorical(values)})
        tm.assert_frame_equal(result2, expected2, check_dtype=True)

    def test_selective_categoricals(self):
        values = ['', '', '', '', '']
        df = pd.DataFrame({'strings': values})
        field = pa.field('strings', pa.string())
        schema = pa.schema([field])
        table = pa.Table.from_pandas(df, schema=schema)
        expected_str = pd.DataFrame({'strings': values})
        expected_cat = pd.DataFrame({'strings': pd.Categorical(values)})
        result1 = table.to_pandas(categories=['strings'])
        tm.assert_frame_equal(result1, expected_cat, check_dtype=True)
        result2 = table.to_pandas(categories=[])
        tm.assert_frame_equal(result2, expected_str, check_dtype=True)
        result3 = table.to_pandas(categories=('strings',))
        tm.assert_frame_equal(result3, expected_cat, check_dtype=True)
        result4 = table.to_pandas(categories=tuple())
        tm.assert_frame_equal(result4, expected_str, check_dtype=True)

    def test_to_pandas_categorical_zero_length(self):
        array = pa.array([], type=pa.int32())
        table = pa.Table.from_arrays(arrays=[array], names=['col'])
        table.to_pandas(categories=['col'])

    def test_to_pandas_categories_already_dictionary(self):
        array = pa.array(['foo', 'foo', 'foo', 'bar']).dictionary_encode()
        table = pa.Table.from_arrays(arrays=[array], names=['col'])
        result = table.to_pandas(categories=['col'])
        assert table.to_pandas().equals(result)

    def test_table_str_to_categorical_without_na(self):
        values = ['a', 'a', 'b', 'b', 'c']
        df = pd.DataFrame({'strings': values})
        field = pa.field('strings', pa.string())
        schema = pa.schema([field])
        table = pa.Table.from_pandas(df, schema=schema)
        result = table.to_pandas(strings_to_categorical=True)
        expected = pd.DataFrame({'strings': pd.Categorical(values)})
        tm.assert_frame_equal(result, expected, check_dtype=True)
        with pytest.raises(pa.ArrowInvalid):
            table.to_pandas(strings_to_categorical=True, zero_copy_only=True)

    def test_table_str_to_categorical_with_na(self):
        values = [None, 'a', 'b', np.nan]
        df = pd.DataFrame({'strings': values})
        field = pa.field('strings', pa.string())
        schema = pa.schema([field])
        table = pa.Table.from_pandas(df, schema=schema)
        result = table.to_pandas(strings_to_categorical=True)
        expected = pd.DataFrame({'strings': pd.Categorical(values)})
        tm.assert_frame_equal(result, expected, check_dtype=True)
        with pytest.raises(pa.ArrowInvalid):
            table.to_pandas(strings_to_categorical=True, zero_copy_only=True)

    def test_array_of_bytes_to_strings(self):
        converted = pa.array(np.array([b'x'], dtype=object), pa.string())
        assert converted.type == pa.string()

    def test_array_of_bytes_to_strings_bad_data(self):
        with pytest.raises(pa.lib.ArrowInvalid, match='was not a utf8 string'):
            pa.array(np.array([b'\x80\x81'], dtype=object), pa.string())

    def test_numpy_string_array_to_fixed_size_binary(self):
        arr = np.array([b'foo', b'bar', b'baz'], dtype='|S3')
        converted = pa.array(arr, type=pa.binary(3))
        expected = pa.array(list(arr), type=pa.binary(3))
        assert converted.equals(expected)
        mask = np.array([False, True, False])
        converted = pa.array(arr, type=pa.binary(3), mask=mask)
        expected = pa.array([b'foo', None, b'baz'], type=pa.binary(3))
        assert converted.equals(expected)
        with pytest.raises(pa.lib.ArrowInvalid, match='Got bytestring of length 3 \\(expected 4\\)'):
            arr = np.array([b'foo', b'bar', b'baz'], dtype='|S3')
            pa.array(arr, type=pa.binary(4))
        with pytest.raises(pa.lib.ArrowInvalid, match='Got bytestring of length 12 \\(expected 3\\)'):
            arr = np.array([b'foo', b'bar', b'baz'], dtype='|U3')
            pa.array(arr, type=pa.binary(3))