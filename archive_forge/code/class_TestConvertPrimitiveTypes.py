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
class TestConvertPrimitiveTypes:
    """
    Conversion tests for primitive (e.g. numeric) types.
    """

    def test_float_no_nulls(self):
        data = {}
        fields = []
        dtypes = [('f2', pa.float16()), ('f4', pa.float32()), ('f8', pa.float64())]
        num_values = 100
        for numpy_dtype, arrow_dtype in dtypes:
            values = np.random.randn(num_values)
            data[numpy_dtype] = values.astype(numpy_dtype)
            fields.append(pa.field(numpy_dtype, arrow_dtype))
        df = pd.DataFrame(data)
        schema = pa.schema(fields)
        _check_pandas_roundtrip(df, expected_schema=schema)

    def test_float_nulls(self):
        num_values = 100
        null_mask = np.random.randint(0, 10, size=num_values) < 3
        dtypes = [('f2', pa.float16()), ('f4', pa.float32()), ('f8', pa.float64())]
        names = ['f2', 'f4', 'f8']
        expected_cols = []
        arrays = []
        fields = []
        for name, arrow_dtype in dtypes:
            values = np.random.randn(num_values).astype(name)
            arr = pa.array(values, from_pandas=True, mask=null_mask)
            arrays.append(arr)
            fields.append(pa.field(name, arrow_dtype))
            values[null_mask] = np.nan
            expected_cols.append(values)
        ex_frame = pd.DataFrame(dict(zip(names, expected_cols)), columns=names)
        table = pa.Table.from_arrays(arrays, names)
        assert table.schema.equals(pa.schema(fields))
        result = table.to_pandas()
        tm.assert_frame_equal(result, ex_frame)

    def test_float_nulls_to_ints(self):
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan]})
        schema = pa.schema([pa.field('a', pa.int16(), nullable=True)])
        table = pa.Table.from_pandas(df, schema=schema, safe=False)
        assert table[0].to_pylist() == [1, 2, None]
        tm.assert_frame_equal(df, table.to_pandas())

    def test_float_nulls_to_boolean(self):
        s = pd.Series([0.0, 1.0, 2.0, None, -3.0])
        expected = pd.Series([False, True, True, None, True])
        _check_array_roundtrip(s, expected=expected, type=pa.bool_())

    def test_series_from_pandas_false_respected(self):
        s = pd.Series([0.0, np.nan])
        arr = pa.array(s, from_pandas=False)
        assert arr.null_count == 0
        assert np.isnan(arr[1].as_py())

    def test_integer_no_nulls(self):
        data = OrderedDict()
        fields = []
        numpy_dtypes = [('i1', pa.int8()), ('i2', pa.int16()), ('i4', pa.int32()), ('i8', pa.int64()), ('u1', pa.uint8()), ('u2', pa.uint16()), ('u4', pa.uint32()), ('u8', pa.uint64()), ('longlong', pa.int64()), ('ulonglong', pa.uint64())]
        num_values = 100
        for dtype, arrow_dtype in numpy_dtypes:
            info = np.iinfo(dtype)
            values = np.random.randint(max(info.min, np.iinfo(np.int_).min), min(info.max, np.iinfo(np.int_).max), size=num_values)
            data[dtype] = values.astype(dtype)
            fields.append(pa.field(dtype, arrow_dtype))
        df = pd.DataFrame(data)
        schema = pa.schema(fields)
        _check_pandas_roundtrip(df, expected_schema=schema)

    def test_all_integer_types(self):
        data = OrderedDict()
        numpy_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'byte', 'ubyte', 'short', 'ushort', 'intc', 'uintc', 'int_', 'uint', 'longlong', 'ulonglong']
        for dtype in numpy_dtypes:
            data[dtype] = np.arange(12, dtype=dtype)
        df = pd.DataFrame(data)
        _check_pandas_roundtrip(df)
        for np_arr in data.values():
            arr = pa.array(np_arr)
            assert arr.to_pylist() == np_arr.tolist()

    def test_integer_byteorder(self):
        int_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
        for dt in int_dtypes:
            for order in '=<>':
                data = np.array([1, 2, 42], dtype=order + dt)
                for np_arr in (data, data[::2]):
                    if data.dtype.isnative:
                        arr = pa.array(data)
                        assert arr.to_pylist() == data.tolist()
                    else:
                        with pytest.raises(NotImplementedError):
                            arr = pa.array(data)

    def test_integer_with_nulls(self):
        int_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
        num_values = 100
        null_mask = np.random.randint(0, 10, size=num_values) < 3
        expected_cols = []
        arrays = []
        for name in int_dtypes:
            values = np.random.randint(0, 100, size=num_values)
            arr = pa.array(values, mask=null_mask)
            arrays.append(arr)
            expected = values.astype('f8')
            expected[null_mask] = np.nan
            expected_cols.append(expected)
        ex_frame = pd.DataFrame(dict(zip(int_dtypes, expected_cols)), columns=int_dtypes)
        table = pa.Table.from_arrays(arrays, int_dtypes)
        result = table.to_pandas()
        tm.assert_frame_equal(result, ex_frame)

    def test_array_from_pandas_type_cast(self):
        arr = np.arange(10, dtype='int64')
        target_type = pa.int8()
        result = pa.array(arr, type=target_type)
        expected = pa.array(arr.astype('int8'))
        assert result.equals(expected)

    def test_boolean_no_nulls(self):
        num_values = 100
        np.random.seed(0)
        df = pd.DataFrame({'bools': np.random.randn(num_values) > 0})
        field = pa.field('bools', pa.bool_())
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected_schema=schema)

    def test_boolean_nulls(self):
        num_values = 100
        np.random.seed(0)
        mask = np.random.randint(0, 10, size=num_values) < 3
        values = np.random.randint(0, 10, size=num_values) < 5
        arr = pa.array(values, mask=mask)
        expected = values.astype(object)
        expected[mask] = None
        field = pa.field('bools', pa.bool_())
        schema = pa.schema([field])
        ex_frame = pd.DataFrame({'bools': expected})
        table = pa.Table.from_arrays([arr], ['bools'])
        assert table.schema.equals(schema)
        result = table.to_pandas()
        tm.assert_frame_equal(result, ex_frame)

    def test_boolean_to_int(self):
        s = pd.Series([True, True, False, True, True] * 2)
        expected = pd.Series([1, 1, 0, 1, 1] * 2)
        _check_array_roundtrip(s, expected=expected, type=pa.int64())

    def test_boolean_objects_to_int(self):
        s = pd.Series([True, True, False, True, True] * 2, dtype=object)
        expected = pd.Series([1, 1, 0, 1, 1] * 2)
        expected_msg = 'Expected integer, got bool'
        with pytest.raises(pa.ArrowTypeError, match=expected_msg):
            _check_array_roundtrip(s, expected=expected, type=pa.int64())

    def test_boolean_nulls_to_float(self):
        s = pd.Series([True, True, False, None, True] * 2)
        expected = pd.Series([1.0, 1.0, 0.0, None, 1.0] * 2)
        _check_array_roundtrip(s, expected=expected, type=pa.float64())

    def test_boolean_multiple_columns(self):
        df = pd.DataFrame(np.ones((3, 2), dtype='bool'), columns=['a', 'b'])
        _check_pandas_roundtrip(df)

    def test_float_object_nulls(self):
        arr = np.array([None, 1.5, np.float64(3.5)] * 5, dtype=object)
        df = pd.DataFrame({'floats': arr})
        expected = pd.DataFrame({'floats': pd.to_numeric(arr)})
        field = pa.field('floats', pa.float64())
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected=expected, expected_schema=schema)

    def test_float_with_null_as_integer(self):
        s = pd.Series([np.nan, 1.0, 2.0, np.nan])
        types = [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
        for ty in types:
            result = pa.array(s, type=ty)
            expected = pa.array([None, 1, 2, None], type=ty)
            assert result.equals(expected)
            df = pd.DataFrame({'has_nulls': s})
            schema = pa.schema([pa.field('has_nulls', ty)])
            result = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            assert result[0].chunk(0).equals(expected)

    def test_int_object_nulls(self):
        arr = np.array([None, 1, np.int64(3)] * 5, dtype=object)
        df = pd.DataFrame({'ints': arr})
        expected = pd.DataFrame({'ints': pd.to_numeric(arr)})
        field = pa.field('ints', pa.int64())
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected=expected, expected_schema=schema)

    def test_boolean_object_nulls(self):
        arr = np.array([False, None, True] * 100, dtype=object)
        df = pd.DataFrame({'bools': arr})
        field = pa.field('bools', pa.bool_())
        schema = pa.schema([field])
        _check_pandas_roundtrip(df, expected_schema=schema)

    def test_all_nulls_cast_numeric(self):
        arr = np.array([None], dtype=object)

        def _check_type(t):
            a2 = pa.array(arr, type=t)
            assert a2.type == t
            assert a2[0].as_py() is None
        _check_type(pa.int32())
        _check_type(pa.float64())

    def test_half_floats_from_numpy(self):
        arr = np.array([1.5, np.nan], dtype=np.float16)
        a = pa.array(arr, type=pa.float16())
        x, y = a.to_pylist()
        assert isinstance(x, np.float16)
        assert x == 1.5
        assert isinstance(y, np.float16)
        assert np.isnan(y)
        a = pa.array(arr, type=pa.float16(), from_pandas=True)
        x, y = a.to_pylist()
        assert isinstance(x, np.float16)
        assert x == 1.5
        assert y is None