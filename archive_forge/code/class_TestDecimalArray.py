from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
class TestDecimalArray(base.ExtensionTests):

    def _get_expected_exception(self, op_name: str, obj, other) -> type[Exception] | None:
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if op_name == 'count':
            return super().check_reduce(ser, op_name, skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(np.asarray(ser), op_name)()
            tm.assert_almost_equal(result, expected)

    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in ['kurt', 'skew', 'sem', 'median']:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        op_name = all_numeric_reductions
        if op_name in ['skew', 'median']:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)

    def test_compare_array(self, data, comparison_op):
        ser = pd.Series(data)
        alter = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        other = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(ser, data, comparison_op, other)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        context = decimal.getcontext()
        divbyzerotrap = context.traps[decimal.DivisionByZero]
        invalidoptrap = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0
        other = pd.Series([int(d * 100) for d in data])
        self.check_opname(ser, op_name, other)
        if 'mod' not in op_name:
            self.check_opname(ser, op_name, ser * 2)
        self.check_opname(ser, op_name, 0)
        self.check_opname(ser, op_name, 5)
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    def test_fillna_frame(self, data_missing):
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            super().test_fillna_frame(data_missing)

    def test_fillna_limit_pad(self, data_missing):
        msg = "ExtensionArray.fillna 'method' keyword is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            super().test_fillna_limit_pad(data_missing)
        msg = "The 'method' keyword in DecimalArray.fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            super().test_fillna_limit_pad(data_missing)

    @pytest.mark.parametrize('limit_area, input_ilocs, expected_ilocs', [('outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]), ('outside', [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]), ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]), ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]), ('inside', [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]), ('inside', [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]), ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]), ('inside', [0, 1, 0, 1, 0], [0, 1, 1, 1, 0])])
    def test_ffill_limit_area(self, data_missing, limit_area, input_ilocs, expected_ilocs):
        msg = "ExtensionArray.fillna 'method' keyword is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            msg = 'DecimalArray does not implement limit_area'
            with pytest.raises(NotImplementedError, match=msg):
                super().test_ffill_limit_area(data_missing, limit_area, input_ilocs, expected_ilocs)

    def test_fillna_limit_backfill(self, data_missing):
        msg = "Series.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            super().test_fillna_limit_backfill(data_missing)
        msg = "ExtensionArray.fillna 'method' keyword is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            super().test_fillna_limit_backfill(data_missing)
        msg = "The 'method' keyword in DecimalArray.fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            super().test_fillna_limit_backfill(data_missing)

    def test_fillna_no_op_returns_copy(self, data):
        msg = '|'.join(["ExtensionArray.fillna 'method' keyword is deprecated", "The 'method' keyword in DecimalArray.fillna is deprecated"])
        with tm.assert_produces_warning((FutureWarning, DeprecationWarning), match=msg, check_stacklevel=False):
            super().test_fillna_no_op_returns_copy(data)

    def test_fillna_series(self, data_missing):
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            super().test_fillna_series(data_missing)

    def test_fillna_series_method(self, data_missing, fillna_method):
        msg = '|'.join(["ExtensionArray.fillna 'method' keyword is deprecated", "The 'method' keyword in DecimalArray.fillna is deprecated"])
        with tm.assert_produces_warning((FutureWarning, DeprecationWarning), match=msg, check_stacklevel=False):
            super().test_fillna_series_method(data_missing, fillna_method)

    def test_fillna_copy_frame(self, data_missing, using_copy_on_write):
        warn = DeprecationWarning if not using_copy_on_write else None
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
            super().test_fillna_copy_frame(data_missing)

    def test_fillna_copy_series(self, data_missing, using_copy_on_write):
        warn = DeprecationWarning if not using_copy_on_write else None
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
            super().test_fillna_copy_series(data_missing)

    @pytest.mark.parametrize('dropna', [True, False])
    def test_value_counts(self, all_data, dropna, request):
        all_data = all_data[:10]
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            other = all_data
        vcs = pd.Series(all_data).value_counts(dropna=dropna)
        vcs_ex = pd.Series(other).value_counts(dropna=dropna)
        with decimal.localcontext() as ctx:
            ctx.traps[decimal.InvalidOperation] = False
            result = vcs.sort_index()
            expected = vcs_ex.sort_index()
        tm.assert_series_equal(result, expected)

    def test_series_repr(self, data):
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)
        assert 'Decimal: ' in repr(ser)

    @pytest.mark.xfail(reason='Inconsistent array-vs-scalar behavior')
    @pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)