from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
class TestAstypeAPI:

    def test_astype_unitless_dt64_raises(self):
        ser = Series(['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]')
        df = ser.to_frame()
        msg = "Casting to unit-less dtype 'datetime64' is not supported"
        with pytest.raises(TypeError, match=msg):
            ser.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            df.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            ser.astype('datetime64')
        with pytest.raises(TypeError, match=msg):
            df.astype('datetime64')

    def test_arg_for_errors_in_astype(self):
        ser = Series([1, 2, 3])
        msg = "Expected value of kwarg 'errors' to be one of \\['raise', 'ignore'\\]\\. Supplied value is 'False'"
        with pytest.raises(ValueError, match=msg):
            ser.astype(np.float64, errors=False)
        ser.astype(np.int8, errors='raise')

    @pytest.mark.parametrize('dtype_class', [dict, Series])
    def test_astype_dict_like(self, dtype_class):
        ser = Series(range(0, 10, 2), name='abc')
        dt1 = dtype_class({'abc': str})
        result = ser.astype(dt1)
        expected = Series(['0', '2', '4', '6', '8'], name='abc', dtype=object)
        tm.assert_series_equal(result, expected)
        dt2 = dtype_class({'abc': 'float64'})
        result = ser.astype(dt2)
        expected = Series([0.0, 2.0, 4.0, 6.0, 8.0], dtype='float64', name='abc')
        tm.assert_series_equal(result, expected)
        dt3 = dtype_class({'abc': str, 'def': str})
        msg = 'Only the Series name can be used for the key in Series dtype mappings\\.'
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt3)
        dt4 = dtype_class({0: str})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt4)
        if dtype_class is Series:
            dt5 = dtype_class({}, dtype=object)
        else:
            dt5 = dtype_class({})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt5)