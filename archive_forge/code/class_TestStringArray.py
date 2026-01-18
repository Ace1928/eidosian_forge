from __future__ import annotations
import string
from typing import cast
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base
class TestStringArray(base.ExtensionTests):

    def test_eq_with_str(self, dtype):
        assert dtype == f'string[{dtype.storage}]'
        super().test_eq_with_str(dtype)

    def test_is_not_string_type(self, dtype):
        assert is_string_dtype(dtype)

    def test_view(self, data, request, arrow_string_storage):
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason='2D support not implemented for ArrowStringArray')
        super().test_view(data)

    def test_from_dtype(self, data):
        pass

    def test_transpose(self, data, request, arrow_string_storage):
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason='2D support not implemented for ArrowStringArray')
        super().test_transpose(data)

    def test_setitem_preserves_views(self, data, request, arrow_string_storage):
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason='2D support not implemented for ArrowStringArray')
        super().test_setitem_preserves_views(data)

    def test_dropna_array(self, data_missing):
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]
        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)
        result = data.fillna(method='backfill')
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def _get_expected_exception(self, op_name: str, obj, other) -> type[Exception] | None:
        if op_name in ['__divmod__', '__rdivmod__']:
            if isinstance(obj, pd.Series) and cast(StringDtype, tm.get_dtype(obj)).storage in ['pyarrow', 'pyarrow_numpy']:
                return NotImplementedError
            elif isinstance(other, pd.Series) and cast(StringDtype, tm.get_dtype(other)).storage in ['pyarrow', 'pyarrow_numpy']:
                return NotImplementedError
            return TypeError
        elif op_name in ['__mod__', '__rmod__', '__pow__', '__rpow__']:
            if cast(StringDtype, tm.get_dtype(obj)).storage in ['pyarrow', 'pyarrow_numpy']:
                return NotImplementedError
            return TypeError
        elif op_name in ['__mul__', '__rmul__']:
            return TypeError
        elif op_name in ['__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__', '__sub__', '__rsub__']:
            if cast(StringDtype, tm.get_dtype(obj)).storage in ['pyarrow', 'pyarrow_numpy']:
                import pyarrow as pa
                return pa.ArrowNotImplementedError
            return TypeError
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return op_name in ['min', 'max'] or (ser.dtype.storage == 'pyarrow_numpy' and op_name in ('any', 'all'))

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        dtype = cast(StringDtype, tm.get_dtype(obj))
        if op_name in ['__add__', '__radd__']:
            cast_to = dtype
        elif dtype.storage == 'pyarrow':
            cast_to = 'boolean[pyarrow]'
        elif dtype.storage == 'pyarrow_numpy':
            cast_to = np.bool_
        else:
            cast_to = 'boolean'
        return pointwise_result.astype(cast_to)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 'abc')

    @pytest.mark.filterwarnings('ignore:Falling back:pandas.errors.PerformanceWarning')
    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)