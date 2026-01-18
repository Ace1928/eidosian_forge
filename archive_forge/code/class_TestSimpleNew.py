import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
class TestSimpleNew:

    def test_constructor_simple_new(self):
        idx = period_range('2007-01', name='p', periods=2, freq='M')
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name='p')
        result = idx._simple_new(idx._data, name='p')
        tm.assert_index_equal(result, idx)
        msg = 'Should be numpy array of type i8'
        with pytest.raises(AssertionError, match=msg):
            type(idx._data)._simple_new(Index(idx.asi8), dtype=idx.dtype)
        arr = type(idx._data)._simple_new(idx.asi8, dtype=idx.dtype)
        result = idx._simple_new(arr, name='p')
        tm.assert_index_equal(result, idx)

    def test_constructor_simple_new_empty(self):
        idx = PeriodIndex([], freq='M', name='p')
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name='p')
        result = idx._simple_new(idx._data, name='p')
        tm.assert_index_equal(result, idx)

    @pytest.mark.parametrize('floats', [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_period_index_simple_new_disallows_floats(self, floats):
        with pytest.raises(AssertionError, match='<class '):
            PeriodIndex._simple_new(floats)