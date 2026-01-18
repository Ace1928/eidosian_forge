from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
class ConstructorTests:
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """

    @pytest.fixture(params=[([3, 14, 15, 92, 653], np.int64), (np.arange(10, dtype='int64'), np.int64), (Index(np.arange(-10, 11, dtype=np.int64)), np.int64), (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64), (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64), (date_range('20180101', periods=10), '<M8[ns]'), (date_range('20180101', periods=10, tz='US/Eastern'), 'datetime64[ns, US/Eastern]'), (timedelta_range('1 day', periods=10), '<m8[ns]')])
    def breaks_and_expected_subtype(self, request):
        return request.param

    def test_constructor(self, constructor, breaks_and_expected_subtype, closed, name):
        breaks, expected_subtype = breaks_and_expected_subtype
        result_kwargs = self.get_kwargs_from_breaks(breaks, closed)
        result = constructor(closed=closed, name=name, **result_kwargs)
        assert result.closed == closed
        assert result.name == name
        assert result.dtype.subtype == expected_subtype
        tm.assert_index_equal(result.left, Index(breaks[:-1], dtype=expected_subtype))
        tm.assert_index_equal(result.right, Index(breaks[1:], dtype=expected_subtype))

    @pytest.mark.parametrize('breaks, subtype', [(Index([0, 1, 2, 3, 4], dtype=np.int64), 'float64'), (Index([0, 1, 2, 3, 4], dtype=np.int64), 'datetime64[ns]'), (Index([0, 1, 2, 3, 4], dtype=np.int64), 'timedelta64[ns]'), (Index([0, 1, 2, 3, 4], dtype=np.float64), 'int64'), (date_range('2017-01-01', periods=5), 'int64'), (timedelta_range('1 day', periods=5), 'int64')])
    def test_constructor_dtype(self, constructor, breaks, subtype):
        expected_kwargs = self.get_kwargs_from_breaks(breaks.astype(subtype))
        expected = constructor(**expected_kwargs)
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        iv_dtype = IntervalDtype(subtype, 'right')
        for dtype in (iv_dtype, str(iv_dtype)):
            result = constructor(dtype=dtype, **result_kwargs)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('breaks', [Index([0, 1, 2, 3, 4], dtype=np.int64), Index([0, 1, 2, 3, 4], dtype=np.uint64), Index([0, 1, 2, 3, 4], dtype=np.float64), date_range('2017-01-01', periods=5), timedelta_range('1 day', periods=5)])
    def test_constructor_pass_closed(self, constructor, breaks):
        iv_dtype = IntervalDtype(breaks.dtype)
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        for dtype in (iv_dtype, str(iv_dtype)):
            with tm.assert_produces_warning(None):
                result = constructor(dtype=dtype, closed='left', **result_kwargs)
            assert result.dtype.closed == 'left'

    @pytest.mark.parametrize('breaks', [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(self, constructor, breaks, closed):
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)
        expected_subtype = np.float64
        expected_values = np.array(breaks[:-1], dtype=object)
        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(np.array(result), expected_values)

    @pytest.mark.parametrize('breaks', [[], np.array([], dtype='int64'), np.array([], dtype='uint64'), np.array([], dtype='float64'), np.array([], dtype='datetime64[ns]'), np.array([], dtype='timedelta64[ns]')])
    def test_constructor_empty(self, constructor, breaks, closed):
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)
        expected_values = np.array([], dtype=object)
        expected_subtype = getattr(breaks, 'dtype', np.int64)
        assert result.empty
        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(np.array(result), expected_values)

    @pytest.mark.parametrize('breaks', [tuple('0123456789'), list('abcdefghij'), np.array(list('abcdefghij'), dtype=object), np.array(list('abcdefghij'), dtype='<U1')])
    def test_constructor_string(self, constructor, breaks):
        msg = 'category, object, and string subtypes are not supported for IntervalIndex'
        with pytest.raises(TypeError, match=msg):
            constructor(**self.get_kwargs_from_breaks(breaks))

    @pytest.mark.parametrize('cat_constructor', [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(self, constructor, cat_constructor):
        breaks = np.arange(10, dtype='int64')
        expected = IntervalIndex.from_breaks(breaks)
        cat_breaks = cat_constructor(breaks)
        result_kwargs = self.get_kwargs_from_breaks(cat_breaks)
        result = constructor(**result_kwargs)
        tm.assert_index_equal(result, expected)

    def test_generic_errors(self, constructor):
        filler = self.get_kwargs_from_breaks(range(10))
        msg = "closed must be one of 'right', 'left', 'both', 'neither'"
        with pytest.raises(ValueError, match=msg):
            constructor(closed='invalid', **filler)
        msg = 'dtype must be an IntervalDtype, got int64'
        with pytest.raises(TypeError, match=msg):
            constructor(dtype='int64', **filler)
        msg = 'data type ["\']invalid["\'] not understood'
        with pytest.raises(TypeError, match=msg):
            constructor(dtype='invalid', **filler)
        periods = period_range('2000-01-01', periods=10)
        periods_kwargs = self.get_kwargs_from_breaks(periods)
        msg = 'Period dtypes are not supported, use a PeriodIndex instead'
        with pytest.raises(ValueError, match=msg):
            constructor(**periods_kwargs)
        decreasing_kwargs = self.get_kwargs_from_breaks(range(10, -1, -1))
        msg = 'left side of interval must be <= right side'
        with pytest.raises(ValueError, match=msg):
            constructor(**decreasing_kwargs)