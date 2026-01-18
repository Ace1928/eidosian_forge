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
class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""

    @pytest.fixture(params=[IntervalIndex, partial(Index, dtype='interval')], ids=['IntervalIndex', 'Index'])
    def klass(self, request):
        return request.param

    @pytest.fixture
    def constructor(self):
        return IntervalIndex

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by the IntervalIndex/Index constructors
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f'{breaks.dtype} not relevant for class constructor tests')
        if len(breaks) == 0:
            return {'data': breaks}
        ivs = [Interval(left, right, closed) if notna(left) else left for left, right in zip(breaks[:-1], breaks[1:])]
        if isinstance(breaks, list):
            return {'data': ivs}
        elif isinstance(getattr(breaks, 'dtype', None), CategoricalDtype):
            return {'data': breaks._constructor(ivs)}
        return {'data': np.array(ivs, dtype=object)}

    def test_generic_errors(self, constructor):
        """
        override the base class implementation since errors are handled
        differently; checks unnecessary since caught at the Interval level
        """

    def test_constructor_string(self):
        pass

    def test_constructor_errors(self, klass):
        ivs = [Interval(0, 1, closed='right'), Interval(2, 3, closed='left')]
        msg = 'intervals must all be closed on the same side'
        with pytest.raises(ValueError, match=msg):
            klass(ivs)
        msg = '(IntervalIndex|Index)\\(...\\) must be called with a collection of some kind, 5 was passed'
        with pytest.raises(TypeError, match=msg):
            klass(5)
        msg = "type <class 'numpy.int(32|64)'> with value 0 is not an interval"
        with pytest.raises(TypeError, match=msg):
            klass([0, 1])

    @pytest.mark.parametrize('data, closed', [([], 'both'), ([np.nan, np.nan], 'neither'), ([Interval(0, 3, closed='neither'), Interval(2, 5, closed='neither')], 'left'), ([Interval(0, 3, closed='left'), Interval(2, 5, closed='right')], 'neither'), (IntervalIndex.from_breaks(range(5), closed='both'), 'right')])
    def test_override_inferred_closed(self, constructor, data, closed):
        if isinstance(data, IntervalIndex):
            tuples = data.to_tuples()
        else:
            tuples = [(iv.left, iv.right) if notna(iv) else iv for iv in data]
        expected = IntervalIndex.from_tuples(tuples, closed=closed)
        result = constructor(data, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('values_constructor', [list, np.array, IntervalIndex, IntervalArray])
    def test_index_object_dtype(self, values_constructor):
        intervals = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
        values = values_constructor(intervals)
        result = Index(values, dtype=object)
        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_index_mixed_closed(self):
        intervals = [Interval(0, 1, closed='left'), Interval(1, 2, closed='right'), Interval(2, 3, closed='neither'), Interval(3, 4, closed='both')]
        result = Index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)