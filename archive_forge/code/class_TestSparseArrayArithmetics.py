import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestSparseArrayArithmetics:

    def _assert(self, a, b):
        tm.assert_numpy_array_equal(a, b)

    def _check_numeric_ops(self, a, b, a_dense, b_dense, mix: bool, op):
        if isinstance(a_dense, np.ndarray):
            expected = op(pd.Series(a_dense), b_dense).values
        elif isinstance(b_dense, np.ndarray):
            expected = op(a_dense, pd.Series(b_dense)).values
        else:
            raise NotImplementedError
        with np.errstate(invalid='ignore', divide='ignore'):
            if mix:
                result = op(a, b_dense).to_dense()
            else:
                result = op(a, b).to_dense()
        self._assert(result, expected)

    def _check_bool_result(self, res):
        assert isinstance(res, SparseArray)
        assert isinstance(res.dtype, SparseDtype)
        assert res.dtype.subtype == np.bool_
        assert isinstance(res.fill_value, bool)

    def _check_comparison_ops(self, a, b, a_dense, b_dense):
        with np.errstate(invalid='ignore'):
            self._check_bool_result(a == b)
            self._assert((a == b).to_dense(), a_dense == b_dense)
            self._check_bool_result(a != b)
            self._assert((a != b).to_dense(), a_dense != b_dense)
            self._check_bool_result(a >= b)
            self._assert((a >= b).to_dense(), a_dense >= b_dense)
            self._check_bool_result(a <= b)
            self._assert((a <= b).to_dense(), a_dense <= b_dense)
            self._check_bool_result(a > b)
            self._assert((a > b).to_dense(), a_dense > b_dense)
            self._check_bool_result(a < b)
            self._assert((a < b).to_dense(), a_dense < b_dense)
            self._check_bool_result(a == b_dense)
            self._assert((a == b_dense).to_dense(), a_dense == b_dense)
            self._check_bool_result(a != b_dense)
            self._assert((a != b_dense).to_dense(), a_dense != b_dense)
            self._check_bool_result(a >= b_dense)
            self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)
            self._check_bool_result(a <= b_dense)
            self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)
            self._check_bool_result(a > b_dense)
            self._assert((a > b_dense).to_dense(), a_dense > b_dense)
            self._check_bool_result(a < b_dense)
            self._assert((a < b_dense).to_dense(), a_dense < b_dense)

    def _check_logical_ops(self, a, b, a_dense, b_dense):
        self._check_bool_result(a & b)
        self._assert((a & b).to_dense(), a_dense & b_dense)
        self._check_bool_result(a | b)
        self._assert((a | b).to_dense(), a_dense | b_dense)
        self._check_bool_result(a & b_dense)
        self._assert((a & b_dense).to_dense(), a_dense & b_dense)
        self._check_bool_result(a | b_dense)
        self._assert((a | b_dense).to_dense(), a_dense | b_dense)

    @pytest.mark.parametrize('scalar', [0, 1, 3])
    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_scalar(self, kind, mix, all_arithmetic_functions, fill_value, scalar, request):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind, fill_value=fill_value)
        self._check_numeric_ops(a, scalar, values, scalar, mix, op)

    def test_float_scalar_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)
        a = SparseArray(values, kind=kind, fill_value=0)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)
        a = SparseArray(values, kind=kind, fill_value=2)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

    def test_float_same_index_without_nans(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions
        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_with_nans(self, kind, mix, all_arithmetic_functions, request):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_float_array(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_different_kind(self, mix, all_arithmetic_functions):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])
        a = SparseArray(values, kind='integer')
        b = SparseArray(rvalues, kind='block')
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, kind='integer', fill_value=0)
        b = SparseArray(rvalues, kind='block')
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind='integer', fill_value=0)
        b = SparseArray(rvalues, kind='block', fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind='integer', fill_value=1)
        b = SparseArray(rvalues, kind='block', fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_int_array(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions
        dtype = np.int64
        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)
        a = SparseArray(values, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype, fill_value=1)
        b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_int_array_comparison(self, kind):
        dtype = 'int64'
        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)
        a = SparseArray(values, dtype=dtype, kind=kind)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=1)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_same_index(self, kind, fill_value):
        values = np.array([True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, True, True], dtype=np.bool_)
        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_array_logical(self, kind, fill_value):
        values = np.array([True, False, True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, False, True, False, True], dtype=np.bool_)
        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    def test_mixed_array_float_int(self, kind, mix, all_arithmetic_functions, request):
        op = all_arithmetic_functions
        rdtype = 'int64'
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_mixed_array_comparison(self, kind):
        rdtype = 'int64'
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_xor(self):
        s = SparseArray([True, True, False, False])
        t = SparseArray([True, False, True, False])
        result = s ^ t
        sp_index = pd.core.arrays.sparse.IntIndex(4, np.array([0, 1, 2], dtype='int32'))
        expected = SparseArray([False, True, True], sparse_index=sp_index)
        tm.assert_sp_array_equal(result, expected)