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
class TestConvertListTypes:
    """
    Conversion tests for list<> types.
    """

    def test_column_of_arrays(self):
        df, schema = dataframe_with_arrays()
        _check_pandas_roundtrip(df, schema=schema, expected_schema=schema)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        expected_schema = schema.with_metadata(table.schema.metadata)
        assert table.schema.equals(expected_schema)
        for column in df.columns:
            field = schema.field(column)
            _check_array_roundtrip(df[column], type=field.type)

    def test_column_of_arrays_to_py(self):
        dtype = 'i1'
        arr = np.array([np.arange(10, dtype=dtype), np.arange(5, dtype=dtype), None, np.arange(1, dtype=dtype)], dtype=object)
        type_ = pa.list_(pa.int8())
        parr = pa.array(arr, type=type_)
        assert parr[0].as_py() == list(range(10))
        assert parr[1].as_py() == list(range(5))
        assert parr[2].as_py() is None
        assert parr[3].as_py() == [0]

    def test_column_of_boolean_list(self):
        array = pa.array([[True, False], [True]], type=pa.list_(pa.bool_()))
        table = pa.Table.from_arrays([array], names=['col1'])
        df = table.to_pandas()
        expected_df = pd.DataFrame({'col1': [[True, False], [True]]})
        tm.assert_frame_equal(df, expected_df)
        s = table[0].to_pandas()
        tm.assert_series_equal(pd.Series(s), df['col1'], check_names=False)

    def test_column_of_decimal_list(self):
        array = pa.array([[decimal.Decimal('1'), decimal.Decimal('2')], [decimal.Decimal('3.3')]], type=pa.list_(pa.decimal128(2, 1)))
        table = pa.Table.from_arrays([array], names=['col1'])
        df = table.to_pandas()
        expected_df = pd.DataFrame({'col1': [[decimal.Decimal('1'), decimal.Decimal('2')], [decimal.Decimal('3.3')]]})
        tm.assert_frame_equal(df, expected_df)

    def test_nested_types_from_ndarray_null_entries(self):
        s = pd.Series(np.array([np.nan, np.nan], dtype=object))
        for ty in [pa.list_(pa.int64()), pa.large_list(pa.int64()), pa.struct([pa.field('f0', 'int32')])]:
            result = pa.array(s, type=ty)
            expected = pa.array([None, None], type=ty)
            assert result.equals(expected)
            with pytest.raises(TypeError):
                pa.array(s.values, type=ty)

    def test_column_of_lists(self):
        df, schema = dataframe_with_lists()
        _check_pandas_roundtrip(df, schema=schema, expected_schema=schema)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        expected_schema = schema.with_metadata(table.schema.metadata)
        assert table.schema.equals(expected_schema)
        for column in df.columns:
            field = schema.field(column)
            _check_array_roundtrip(df[column], type=field.type)

    def test_column_of_lists_first_empty(self):
        num_lists = [[], [2, 3, 4], [3, 6, 7, 8], [], [2]]
        series = pd.Series([np.array(s, dtype=float) for s in num_lists])
        arr = pa.array(series)
        result = pd.Series(arr.to_pandas())
        tm.assert_series_equal(result, series)

    def test_column_of_lists_chunked(self):
        df = pd.DataFrame({'lists': np.array([[1, 2], None, [2, 3], [4, 5], [6, 7], [8, 9]], dtype=object)})
        schema = pa.schema([pa.field('lists', pa.list_(pa.int64()))])
        t1 = pa.Table.from_pandas(df[:2], schema=schema)
        t2 = pa.Table.from_pandas(df[2:], schema=schema)
        table = pa.concat_tables([t1, t2])
        result = table.to_pandas()
        tm.assert_frame_equal(result, df)

    def test_empty_column_of_lists_chunked(self):
        df = pd.DataFrame({'lists': np.array([], dtype=object)})
        schema = pa.schema([pa.field('lists', pa.list_(pa.int64()))])
        table = pa.Table.from_pandas(df, schema=schema)
        result = table.to_pandas()
        tm.assert_frame_equal(result, df)

    def test_column_of_lists_chunked2(self):
        data1 = [[0, 1], [2, 3], [4, 5], [6, 7], [10, 11], [12, 13], [14, 15], [16, 17]]
        data2 = [[8, 9], [18, 19]]
        a1 = pa.array(data1)
        a2 = pa.array(data2)
        t1 = pa.Table.from_arrays([a1], names=['a'])
        t2 = pa.Table.from_arrays([a2], names=['a'])
        concatenated = pa.concat_tables([t1, t2])
        result = concatenated.to_pandas()
        expected = pd.DataFrame({'a': data1 + data2})
        tm.assert_frame_equal(result, expected)

    def test_column_of_lists_strided(self):
        df, schema = dataframe_with_lists()
        df = pd.concat([df] * 6, ignore_index=True)
        arr = df['int64'].values[::3]
        assert arr.strides[0] != 8
        _check_array_roundtrip(arr)

    def test_nested_lists_all_none(self):
        data = np.array([[None, None], None], dtype=object)
        arr = pa.array(data)
        expected = pa.array(list(data))
        assert arr.equals(expected)
        assert arr.type == pa.list_(pa.null())
        data2 = np.array([None, None, [None, None], np.array([None, None], dtype=object)], dtype=object)
        arr = pa.array(data2)
        expected = pa.array([None, None, [None, None], [None, None]])
        assert arr.equals(expected)

    def test_nested_lists_all_empty(self):
        data = pd.Series([[], [], []])
        arr = pa.array(data)
        expected = pa.array(list(data))
        assert arr.equals(expected)
        assert arr.type == pa.list_(pa.null())

    def test_nested_list_first_empty(self):
        data = pd.Series([[], ['a']])
        arr = pa.array(data)
        expected = pa.array(list(data))
        assert arr.equals(expected)
        assert arr.type == pa.list_(pa.string())

    def test_nested_smaller_ints(self):
        data = pd.Series([np.array([1, 2, 3], dtype='i1'), None])
        result = pa.array(data)
        result2 = pa.array(data.values)
        expected = pa.array([[1, 2, 3], None], type=pa.list_(pa.int8()))
        assert result.equals(expected)
        assert result2.equals(expected)
        data3 = pd.Series([np.array([1, 2, 3], dtype='f4'), None])
        result3 = pa.array(data3)
        expected3 = pa.array([[1, 2, 3], None], type=pa.list_(pa.float32()))
        assert result3.equals(expected3)

    def test_infer_lists(self):
        data = OrderedDict([('nan_ints', [[np.nan, 1], [2, 3]]), ('ints', [[0, 1], [2, 3]]), ('strs', [[None, 'b'], ['c', 'd']]), ('nested_strs', [[[None, 'b'], ['c', 'd']], None])])
        df = pd.DataFrame(data)
        expected_schema = pa.schema([pa.field('nan_ints', pa.list_(pa.int64())), pa.field('ints', pa.list_(pa.int64())), pa.field('strs', pa.list_(pa.string())), pa.field('nested_strs', pa.list_(pa.list_(pa.string())))])
        _check_pandas_roundtrip(df, expected_schema=expected_schema)

    def test_fixed_size_list(self):
        fixed_ty = pa.list_(pa.int64(), list_size=4)
        variable_ty = pa.list_(pa.int64())
        data = [[0, 1, 2, 3], None, [4, 5, 6, 7], [8, 9, 10, 11]]
        fixed_arr = pa.array(data, type=fixed_ty)
        variable_arr = pa.array(data, type=variable_ty)
        result = fixed_arr.to_pandas()
        expected = variable_arr.to_pandas()
        for left, right in zip(result, expected):
            if left is None:
                assert right is None
            npt.assert_array_equal(left, right)

    def test_infer_numpy_array(self):
        data = OrderedDict([('ints', [np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)])])
        df = pd.DataFrame(data)
        expected_schema = pa.schema([pa.field('ints', pa.list_(pa.int64()))])
        _check_pandas_roundtrip(df, expected_schema=expected_schema)

    def test_to_list_of_structs_pandas(self):
        ints = pa.array([1, 2, 3], pa.int32())
        strings = pa.array([['a', 'b'], ['c', 'd'], ['e', 'f']], pa.list_(pa.string()))
        structs = pa.StructArray.from_arrays([ints, strings], ['f1', 'f2'])
        data = pa.ListArray.from_arrays([0, 1, 3], structs)
        expected = pd.Series([[{'f1': 1, 'f2': ['a', 'b']}], [{'f1': 2, 'f2': ['c', 'd']}, {'f1': 3, 'f2': ['e', 'f']}]])
        series = pd.Series(data.to_pandas())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
            tm.assert_series_equal(series, expected)

    def test_to_list_of_maps_pandas(self):
        if Version(np.__version__) >= Version('1.25.0.dev0') and Version(pd.__version__) < Version('2.0.0'):
            pytest.skip('Regression in pandas with numpy 1.25')
        data = [[[('foo', ['a', 'b']), ('bar', ['c', 'd'])]], [[('baz', []), ('qux', None), ('quux', [None, 'e'])], [('quz', ['f', 'g'])]]]
        arr = pa.array(data, pa.list_(pa.map_(pa.utf8(), pa.list_(pa.utf8()))))
        series = arr.to_pandas()
        expected = pd.Series(data)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
            tm.assert_series_equal(series, expected)

    def test_to_list_of_maps_pandas_sliced(self):
        """
        A slightly more rigorous test for chunk/slice combinations
        """
        if Version(np.__version__) >= Version('1.25.0.dev0') and Version(pd.__version__) < Version('2.0.0'):
            pytest.skip('Regression in pandas with numpy 1.25')
        keys = pa.array(['ignore', 'foo', 'bar', 'baz', 'qux', 'quux', 'ignore']).slice(1, 5)
        items = pa.array([['ignore'], ['ignore'], ['a', 'b'], ['c', 'd'], [], None, [None, 'e']], pa.list_(pa.string())).slice(2, 5)
        map = pa.MapArray.from_arrays([0, 2, 4], keys, items)
        arr = pa.ListArray.from_arrays([0, 1, 2], map)
        series = arr.to_pandas()
        expected = pd.Series([[[('foo', ['a', 'b']), ('bar', ['c', 'd'])]], [[('baz', []), ('qux', None)]]])
        series_sliced = arr.slice(1, 2).to_pandas()
        expected_sliced = pd.Series([[[('baz', []), ('qux', None)]]])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
            tm.assert_series_equal(series, expected)
            tm.assert_series_equal(series_sliced, expected_sliced)

    @pytest.mark.parametrize('t,data,expected', [(pa.int64, [[1, 2], [3], None], [None, [3], None]), (pa.string, [['aaa', 'bb'], ['c'], None], [None, ['c'], None]), (pa.null, [[None, None], [None], None], [None, [None], None])])
    def test_array_from_pandas_typed_array_with_mask(self, t, data, expected):
        m = np.array([True, False, True])
        s = pd.Series(data)
        result = pa.Array.from_pandas(s, mask=m, type=pa.list_(t()))
        assert pa.Array.from_pandas(expected, type=pa.list_(t())).equals(result)

    def test_empty_list_roundtrip(self):
        empty_list_array = np.empty((3,), dtype=object)
        empty_list_array.fill([])
        df = pd.DataFrame({'a': np.array(['1', '2', '3']), 'b': empty_list_array})
        tbl = pa.Table.from_pandas(df)
        result = tbl.to_pandas()
        tm.assert_frame_equal(result, df)

    def test_array_from_nested_arrays(self):
        df, schema = dataframe_with_arrays()
        for field in schema:
            arr = df[field.name].values
            expected = pa.array(list(arr), type=field.type)
            result = pa.array(arr)
            assert result.type == field.type
            assert result.equals(expected)

    def test_nested_large_list(self):
        s = pa.array([[[1, 2, 3], [4]], None], type=pa.large_list(pa.large_list(pa.int64()))).to_pandas()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Creating an ndarray from ragged nested', _np_VisibleDeprecationWarning)
            warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
            tm.assert_series_equal(s, pd.Series([[[1, 2, 3], [4]], None], dtype=object), check_names=False)

    def test_large_binary_list(self):
        for list_type_factory in (pa.list_, pa.large_list):
            s = pa.array([['aa', 'bb'], None, ['cc'], []], type=list_type_factory(pa.large_binary())).to_pandas()
            tm.assert_series_equal(s, pd.Series([[b'aa', b'bb'], None, [b'cc'], []]), check_names=False)
            s = pa.array([['aa', 'bb'], None, ['cc'], []], type=list_type_factory(pa.large_string())).to_pandas()
            tm.assert_series_equal(s, pd.Series([['aa', 'bb'], None, ['cc'], []]), check_names=False)

    def test_list_of_dictionary(self):
        child = pa.array(['foo', 'bar', None, 'foo']).dictionary_encode()
        arr = pa.ListArray.from_arrays([0, 1, 3, 3, 4], child)
        expected = pd.Series(arr.to_pylist())
        tm.assert_series_equal(arr.to_pandas(), expected)
        arr = arr.take([0, 1, None, 3])
        expected[2] = None
        tm.assert_series_equal(arr.to_pandas(), expected)

    @pytest.mark.large_memory
    def test_auto_chunking_on_list_overflow(self):
        n = 2 ** 21
        df = pd.DataFrame.from_dict({'a': list(np.zeros((n, 2 ** 10), dtype='uint8')), 'b': range(n)})
        table = pa.Table.from_pandas(df)
        table.validate(full=True)
        column_a = table[0]
        assert column_a.num_chunks == 2
        assert len(column_a.chunk(0)) == 2 ** 21 - 1
        assert len(column_a.chunk(1)) == 1

    def test_map_array_roundtrip(self):
        data = [[(b'a', 1), (b'b', 2)], [(b'c', 3)], [(b'd', 4), (b'e', 5), (b'f', 6)], [(b'g', 7)]]
        df = pd.DataFrame({'map': data})
        schema = pa.schema([('map', pa.map_(pa.binary(), pa.int32()))])
        _check_pandas_roundtrip(df, schema=schema)

    def test_map_array_chunked(self):
        data1 = [[(b'a', 1), (b'b', 2)], [(b'c', 3)], [(b'd', 4), (b'e', 5), (b'f', 6)], [(b'g', 7)]]
        data2 = [[(k, v * 2) for k, v in row] for row in data1]
        arr1 = pa.array(data1, type=pa.map_(pa.binary(), pa.int32()))
        arr2 = pa.array(data2, type=pa.map_(pa.binary(), pa.int32()))
        arr = pa.chunked_array([arr1, arr2])
        expected = pd.Series(data1 + data2)
        actual = arr.to_pandas()
        tm.assert_series_equal(actual, expected, check_names=False)

    def test_map_array_with_nulls(self):
        data = [[(b'a', 1), (b'b', 2)], None, [(b'd', 4), (b'e', 5), (b'f', None)], [(b'g', 7)]]
        expected = [[(k, float(v) if v is not None else None) for k, v in row] if row is not None else None for row in data]
        expected = pd.Series(expected)
        arr = pa.array(data, type=pa.map_(pa.binary(), pa.int32()))
        actual = arr.to_pandas()
        tm.assert_series_equal(actual, expected, check_names=False)

    def test_map_array_dictionary_encoded(self):
        offsets = pa.array([0, 3, 5])
        items = pa.array(['a', 'b', 'c', 'a', 'd']).dictionary_encode()
        keys = pa.array(list(range(len(items))))
        arr = pa.MapArray.from_arrays(offsets, keys, items)
        expected = pd.Series([[(0, 'a'), (1, 'b'), (2, 'c')], [(3, 'a'), (4, 'd')]])
        actual = arr.to_pandas()
        tm.assert_series_equal(actual, expected, check_names=False)

    def test_list_no_duplicate_base(self):
        arr = pa.array([[1, 2], [3, 4, 5], None, [6, None], [7, 8]])
        chunked_arr = pa.chunked_array([arr.slice(0, 3), arr.slice(3, 1)])
        np_arr = chunked_arr.to_numpy()
        expected = np.array([[1.0, 2.0], [3.0, 4.0, 5.0], None, [6.0, np.nan]], dtype='object')
        for left, right in zip(np_arr, expected):
            if right is None:
                assert left == right
            else:
                npt.assert_array_equal(left, right)
        expected_base = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]])
        npt.assert_array_equal(np_arr[0].base, expected_base)
        np_arr_sliced = chunked_arr.slice(1, 3).to_numpy()
        expected = np.array([[3, 4, 5], None, [6, np.nan]], dtype='object')
        for left, right in zip(np_arr_sliced, expected):
            if right is None:
                assert left == right
            else:
                npt.assert_array_equal(left, right)
        expected_base = np.array([[3.0, 4.0, 5.0, 6.0, np.nan]])
        npt.assert_array_equal(np_arr_sliced[0].base, expected_base)

    def test_list_values_behind_null(self):
        arr = pa.ListArray.from_arrays(offsets=pa.array([0, 2, 4, 6]), values=pa.array([1, 2, 99, 99, 3, None]), mask=pa.array([False, True, False]))
        np_arr = arr.to_numpy(zero_copy_only=False)
        expected = np.array([[1.0, 2.0], None, [3.0, np.nan]], dtype='object')
        for left, right in zip(np_arr, expected):
            if right is None:
                assert left == right
            else:
                npt.assert_array_equal(left, right)