import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_object_dtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.tests.extension import base
class TestNumpyExtensionArray(base.ExtensionTests):

    @pytest.mark.skip(reason="We don't register our dtype")
    def test_from_dtype(self, data):
        pass

    @skip_nested
    def test_series_constructor_scalar_with_index(self, data, dtype):
        super().test_series_constructor_scalar_with_index(data, dtype)

    def test_check_dtype(self, data, request, using_infer_string):
        if data.dtype.numpy_dtype == 'object':
            request.applymarker(pytest.mark.xfail(reason=f'NumpyExtensionArray expectedly clashes with a NumPy name: {data.dtype.numpy_dtype}'))
        super().test_check_dtype(data)

    def test_is_not_object_type(self, dtype, request):
        if dtype.numpy_dtype == 'object':
            assert is_object_dtype(dtype)
        else:
            super().test_is_not_object_type(dtype)

    @skip_nested
    def test_getitem_scalar(self, data):
        super().test_getitem_scalar(data)

    @skip_nested
    def test_shift_fill_value(self, data):
        super().test_shift_fill_value(data)

    @skip_nested
    def test_fillna_copy_frame(self, data_missing):
        super().test_fillna_copy_frame(data_missing)

    @skip_nested
    def test_fillna_copy_series(self, data_missing):
        super().test_fillna_copy_series(data_missing)

    @skip_nested
    def test_searchsorted(self, data_for_sorting, as_series):
        super().test_searchsorted(data_for_sorting, as_series)

    @pytest.mark.xfail(reason='NumpyExtensionArray.diff may fail on dtype')
    def test_diff(self, data, periods):
        return super().test_diff(data, periods)

    def test_insert(self, data, request):
        if data.dtype.numpy_dtype == object:
            mark = pytest.mark.xfail(reason='Dimension mismatch in np.concatenate')
            request.applymarker(mark)
        super().test_insert(data)

    @skip_nested
    def test_insert_invalid(self, data, invalid_scalar):
        super().test_insert_invalid(data, invalid_scalar)
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def test_divmod(self, data):
        divmod_exc = None
        if data.dtype.kind == 'O':
            divmod_exc = TypeError
        self.divmod_exc = divmod_exc
        super().test_divmod(data)

    def test_divmod_series_array(self, data):
        ser = pd.Series(data)
        exc = None
        if data.dtype.kind == 'O':
            exc = TypeError
            self.divmod_exc = exc
        self._check_divmod_op(ser, divmod, data)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        opname = all_arithmetic_operators
        series_scalar_exc = None
        if data.dtype.numpy_dtype == object:
            if opname in ['__mul__', '__rmul__']:
                mark = pytest.mark.xfail(reason='the Series.combine step raises but not the Series method.')
                request.node.add_marker(mark)
            series_scalar_exc = TypeError
        self.series_scalar_exc = series_scalar_exc
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        opname = all_arithmetic_operators
        series_array_exc = None
        if data.dtype.numpy_dtype == object and opname not in ['__add__', '__radd__']:
            series_array_exc = TypeError
        self.series_array_exc = series_array_exc
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        opname = all_arithmetic_operators
        frame_scalar_exc = None
        if data.dtype.numpy_dtype == object:
            if opname in ['__mul__', '__rmul__']:
                mark = pytest.mark.xfail(reason='the Series.combine step raises but not the Series method.')
                request.node.add_marker(mark)
            frame_scalar_exc = TypeError
        self.frame_scalar_exc = frame_scalar_exc
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if ser.dtype.kind == 'O':
            return op_name in ['sum', 'min', 'max', 'any', 'all']
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        res_op = getattr(ser, op_name)
        cmp_dtype = ser.dtype.numpy_dtype
        alt = ser.astype(cmp_dtype)
        exp_op = getattr(alt, op_name)
        if op_name == 'count':
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.skip('TODO: tests not written yet')
    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        pass

    @skip_nested
    def test_fillna_series(self, data_missing):
        super().test_fillna_series(data_missing)

    @skip_nested
    def test_fillna_frame(self, data_missing):
        super().test_fillna_frame(data_missing)

    @skip_nested
    def test_setitem_invalid(self, data, invalid_scalar):
        super().test_setitem_invalid(data, invalid_scalar)

    @skip_nested
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    @skip_nested
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(self, data, setter):
        super().test_setitem_mask_broadcast(data, setter)

    @skip_nested
    def test_setitem_scalar_key_sequence_raise(self, data):
        super().test_setitem_scalar_key_sequence_raise(data)

    @skip_nested
    @pytest.mark.parametrize('mask', [np.array([True, True, True, False, False]), pd.array([True, True, True, False, False], dtype='boolean')], ids=['numpy-array', 'boolean-array'])
    def test_setitem_mask(self, data, mask, box_in_series):
        super().test_setitem_mask(data, mask, box_in_series)

    @skip_nested
    @pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
    def test_setitem_integer_array(self, data, idx, box_in_series):
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.parametrize('idx, box_in_series', [([0, 1, 2, pd.NA], False), pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False)], ids=['list-False', 'list-True', 'integer-array-False', 'integer-array-True'])
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @skip_nested
    def test_setitem_slice(self, data, box_in_series):
        super().test_setitem_slice(data, box_in_series)

    @skip_nested
    def test_setitem_loc_iloc_slice(self, data):
        super().test_setitem_loc_iloc_slice(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        df = expected = pd.DataFrame({'data': pd.Series(data)})
        result = pd.DataFrame(index=df.index)
        key = full_indexer(df)
        result.loc[key, 'data'] = df['data']
        if data.dtype.numpy_dtype != object:
            if not isinstance(key, slice) or key != slice(None):
                expected = pd.DataFrame({'data': data.to_numpy()})
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.xfail(reason='NumpyEADtype is unpacked')
    def test_index_from_listlike_with_dtype(self, data):
        super().test_index_from_listlike_with_dtype(data)

    @skip_nested
    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(self, engine, data, request):
        super().test_EA_types(engine, data, request)