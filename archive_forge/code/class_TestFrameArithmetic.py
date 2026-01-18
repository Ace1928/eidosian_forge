from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
class TestFrameArithmetic:

    def test_td64_op_nat_casting(self):
        ser = Series(['NaT', 'NaT'], dtype='timedelta64[ns]')
        df = DataFrame([[1, 2], [3, 4]])
        result = df * ser
        expected = DataFrame({0: ser, 1: ser})
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_rowlike_broadcasts(self):
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        rowlike = arr[[1], :]
        assert rowlike.shape == (1, df.shape[1])
        expected = DataFrame([[2, 4], [4, 6], [6, 8]], columns=df.columns, index=df.index, dtype=arr.dtype)
        result = df + rowlike
        tm.assert_frame_equal(result, expected)
        result = rowlike + df
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_collike_broadcasts(self):
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        collike = arr[:, [1]]
        assert collike.shape == (df.shape[0], 1)
        expected = DataFrame([[1, 2], [5, 6], [9, 10]], columns=df.columns, index=df.index, dtype=arr.dtype)
        result = df + collike
        tm.assert_frame_equal(result, expected)
        result = collike + df
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_rowlike_broadcasts(self, request, all_arithmetic_operators, using_array_manager):
        opname = all_arithmetic_operators
        if using_array_manager and opname in ('__rmod__', '__rfloordiv__'):
            td.mark_array_manager_not_yet_implemented(request)
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        rowlike = arr[[1], :]
        assert rowlike.shape == (1, df.shape[1])
        exvals = [getattr(df.loc['A'], opname)(rowlike.squeeze()), getattr(df.loc['B'], opname)(rowlike.squeeze()), getattr(df.loc['C'], opname)(rowlike.squeeze())]
        expected = DataFrame(exvals, columns=df.columns, index=df.index)
        result = getattr(df, opname)(rowlike)
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_collike_broadcasts(self, request, all_arithmetic_operators, using_array_manager):
        opname = all_arithmetic_operators
        if using_array_manager and opname in ('__rmod__', '__rfloordiv__'):
            td.mark_array_manager_not_yet_implemented(request)
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        collike = arr[:, [1]]
        assert collike.shape == (df.shape[0], 1)
        exvals = {True: getattr(df[True], opname)(collike.squeeze()), False: getattr(df[False], opname)(collike.squeeze())}
        dtype = None
        if opname in ['__rmod__', '__rfloordiv__']:
            dtype = np.common_type(*(x.values for x in exvals.values()))
        expected = DataFrame(exvals, columns=df.columns, index=df.index, dtype=dtype)
        result = getattr(df, opname)(collike)
        tm.assert_frame_equal(result, expected)

    def test_df_bool_mul_int(self):
        df = DataFrame([[False, True], [False, False]])
        result = df * 1
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == 'i').all()
        result = 1 * df
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == 'i').all()

    def test_arith_mixed(self):
        left = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
        result = left + left
        expected = DataFrame({'A': ['aa', 'bb', 'cc'], 'B': [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('col', ['A', 'B'])
    def test_arith_getitem_commute(self, all_arithmetic_functions, col):
        df = DataFrame({'A': [1.1, 3.3], 'B': [2.5, -3.9]})
        result = all_arithmetic_functions(df, 1)[col]
        expected = all_arithmetic_functions(df[col], 1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values', [[1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])])
    def test_arith_alignment_non_pandas_object(self, values):
        df = DataFrame({'A': [1, 1], 'B': [1, 1]})
        expected = DataFrame({'A': [2, 2], 'B': [3, 3]})
        result = df + values
        tm.assert_frame_equal(result, expected)

    def test_arith_non_pandas_object(self):
        df = DataFrame(np.arange(1, 10, dtype='f8').reshape(3, 3), columns=['one', 'two', 'three'], index=['a', 'b', 'c'])
        val1 = df.xs('a').values
        added = DataFrame(df.values + val1, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val1, added)
        added = DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val1, axis=0), added)
        val2 = list(df['two'])
        added = DataFrame(df.values + val2, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val2, added)
        added = DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val2, axis='index'), added)
        val3 = np.random.default_rng(2).random(df.shape)
        added = DataFrame(df.values + val3, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val3), added)

    def test_operations_with_interval_categories_index(self, all_arithmetic_operators):
        op = all_arithmetic_operators
        ind = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
        data = [1, 2]
        df = DataFrame([data], columns=ind)
        num = 10
        result = getattr(df, op)(num)
        expected = DataFrame([[getattr(n, op)(num) for n in data]], columns=ind)
        tm.assert_frame_equal(result, expected)

    def test_frame_with_frame_reindex(self):
        df = DataFrame({'foo': [pd.Timestamp('2019'), pd.Timestamp('2020')], 'bar': [pd.Timestamp('2018'), pd.Timestamp('2021')]}, columns=['foo', 'bar'], dtype='M8[ns]')
        df2 = df[['foo']]
        result = df - df2
        expected = DataFrame({'foo': [pd.Timedelta(0), pd.Timedelta(0)], 'bar': [np.nan, np.nan]}, columns=['bar', 'foo'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value, dtype', [(1, 'i8'), (1.0, 'f8'), (2 ** 63, 'f8'), (1j, 'complex128'), (2 ** 63, 'complex128'), (True, 'bool'), (np.timedelta64(20, 'ns'), '<m8[ns]'), (np.datetime64(20, 'ns'), '<M8[ns]')])
    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv, operator.mod, operator.pow], ids=lambda x: x.__name__)
    def test_binop_other(self, op, value, dtype, switch_numexpr_min_elements):
        skip = {(operator.truediv, 'bool'), (operator.pow, 'bool'), (operator.add, 'bool'), (operator.mul, 'bool')}
        elem = DummyElement(value, dtype)
        df = DataFrame({'A': [elem.value, elem.value]}, dtype=elem.dtype)
        invalid = {(operator.pow, '<M8[ns]'), (operator.mod, '<M8[ns]'), (operator.truediv, '<M8[ns]'), (operator.mul, '<M8[ns]'), (operator.add, '<M8[ns]'), (operator.pow, '<m8[ns]'), (operator.mul, '<m8[ns]'), (operator.sub, 'bool'), (operator.mod, 'complex128')}
        if (op, dtype) in invalid:
            warn = None
            if dtype == '<M8[ns]' and op == operator.add or (dtype == '<m8[ns]' and op == operator.mul):
                msg = None
            elif dtype == 'complex128':
                msg = "ufunc 'remainder' not supported for the input types"
            elif op is operator.sub:
                msg = 'numpy boolean subtract, the `-` operator, is '
                if dtype == 'bool' and expr.USE_NUMEXPR and (switch_numexpr_min_elements == 0):
                    warn = UserWarning
            else:
                msg = f'cannot perform __{op.__name__}__ with this index type: (DatetimeArray|TimedeltaArray)'
            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(warn):
                    op(df, elem.value)
        elif (op, dtype) in skip:
            if op in [operator.add, operator.mul]:
                if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                    warn = UserWarning
                else:
                    warn = None
                with tm.assert_produces_warning(warn):
                    op(df, elem.value)
            else:
                msg = "operator '.*' not implemented for .* dtypes"
                with pytest.raises(NotImplementedError, match=msg):
                    op(df, elem.value)
        else:
            with tm.assert_produces_warning(None):
                result = op(df, elem.value).dtypes
                expected = op(df, value).dtypes
            tm.assert_series_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes(self):
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2 = MultiIndex.from_arrays([Series([1, 2], dtype='Int8'), Series([3, 4])])
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result = left - right
        expected = DataFrame([[0, 0], [0, 0]], columns=midx)
        tm.assert_frame_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes_different_order(self):
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2 = MultiIndex.from_arrays([Series([2, 1], dtype='Int8'), Series([4, 3])])
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result = left - right
        expected = DataFrame([[-1, 1], [-1, 1]], columns=midx)
        tm.assert_frame_equal(result, expected)