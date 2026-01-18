import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
class TestNumpyJSONTests:

    @pytest.mark.parametrize('bool_input', [True, False])
    def test_bool(self, bool_input):
        b = bool(bool_input)
        assert ujson.ujson_loads(ujson.ujson_dumps(b)) == b

    def test_bool_array(self):
        bool_array = np.array([True, False, True, True, False, True, False, False], dtype=bool)
        output = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
        tm.assert_numpy_array_equal(bool_array, output)

    def test_int(self, any_int_numpy_dtype):
        klass = np.dtype(any_int_numpy_dtype).type
        num = klass(1)
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_int_array(self, any_int_numpy_dtype):
        arr = np.arange(100, dtype=int)
        arr_input = arr.astype(any_int_numpy_dtype)
        arr_output = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr_input)), dtype=any_int_numpy_dtype)
        tm.assert_numpy_array_equal(arr_input, arr_output)

    def test_int_max(self, any_int_numpy_dtype):
        if any_int_numpy_dtype in ('int64', 'uint64') and (not IS64):
            pytest.skip('Cannot test 64-bit integer on 32-bit platform')
        klass = np.dtype(any_int_numpy_dtype).type
        if any_int_numpy_dtype == 'uint64':
            num = np.iinfo('int64').max
        else:
            num = np.iinfo(any_int_numpy_dtype).max
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float(self, float_numpy_dtype):
        klass = np.dtype(float_numpy_dtype).type
        num = klass(256.2013)
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float_array(self, float_numpy_dtype):
        arr = np.arange(12.5, 185.72, 1.7322, dtype=float)
        float_input = arr.astype(float_numpy_dtype)
        float_output = np.array(ujson.ujson_loads(ujson.ujson_dumps(float_input, double_precision=15)), dtype=float_numpy_dtype)
        tm.assert_almost_equal(float_input, float_output)

    def test_float_max(self, float_numpy_dtype):
        klass = np.dtype(float_numpy_dtype).type
        num = klass(np.finfo(float_numpy_dtype).max / 10)
        tm.assert_almost_equal(klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num)

    def test_array_basic(self):
        arr = np.arange(96)
        arr = arr.reshape((2, 2, 2, 2, 3, 2))
        tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)

    @pytest.mark.parametrize('shape', [(10, 10), (5, 5, 4), (100, 1)])
    def test_array_reshaped(self, shape):
        arr = np.arange(100)
        arr = arr.reshape(shape)
        tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)

    def test_array_list(self):
        arr_list = ['a', [], {}, {}, [], 42, 97.8, ['a', 'b'], {'key': 'val'}]
        arr = np.array(arr_list, dtype=object)
        result = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
        tm.assert_numpy_array_equal(result, arr)

    def test_array_float(self):
        dtype = np.float32
        arr = np.arange(100.202, 200.202, 1, dtype=dtype)
        arr = arr.reshape((5, 5, 4))
        arr_out = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
        tm.assert_almost_equal(arr, arr_out)

    def test_0d_array(self):
        msg = re.escape('array(1) (numpy-scalar) is not JSON serializable at the moment')
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.array(1))

    def test_array_long_double(self):
        msg = re.compile('1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment')
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.longdouble(1234.5))