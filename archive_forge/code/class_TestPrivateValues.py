import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
class TestPrivateValues:

    @td.skip_array_manager_invalid_test
    def test_private_values_dt64tz(self, using_copy_on_write):
        dta = date_range('2000', periods=4, tz='US/Central')._data.reshape(-1, 1)
        df = DataFrame(dta, columns=['A'])
        tm.assert_equal(df._values, dta)
        if using_copy_on_write:
            assert not np.shares_memory(df._values._ndarray, dta._ndarray)
        else:
            assert np.shares_memory(df._values._ndarray, dta._ndarray)
        tda = dta - dta
        df2 = df - df
        tm.assert_equal(df2._values, tda)

    @td.skip_array_manager_invalid_test
    def test_private_values_dt64tz_multicol(self, using_copy_on_write):
        dta = date_range('2000', periods=8, tz='US/Central')._data.reshape(-1, 2)
        df = DataFrame(dta, columns=['A', 'B'])
        tm.assert_equal(df._values, dta)
        if using_copy_on_write:
            assert not np.shares_memory(df._values._ndarray, dta._ndarray)
        else:
            assert np.shares_memory(df._values._ndarray, dta._ndarray)
        tda = dta - dta
        df2 = df - df
        tm.assert_equal(df2._values, tda)

    def test_private_values_dt64_multiblock(self):
        dta = date_range('2000', periods=8)._data
        df = DataFrame({'A': dta[:4]}, copy=False)
        df['B'] = dta[4:]
        assert len(df._mgr.arrays) == 2
        result = df._values
        expected = dta.reshape(2, 4).T
        tm.assert_equal(result, expected)