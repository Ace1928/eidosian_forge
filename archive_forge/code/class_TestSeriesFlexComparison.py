from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestSeriesFlexComparison:

    @pytest.mark.parametrize('axis', [0, None, 'index'])
    def test_comparison_flex_basic(self, axis, comparison_op):
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        result = getattr(left, comparison_op.__name__)(right, axis=axis)
        expected = comparison_op(left, right)
        tm.assert_series_equal(result, expected)

    def test_comparison_bad_axis(self, comparison_op):
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        msg = 'No axis named 1 for object type'
        with pytest.raises(ValueError, match=msg):
            getattr(left, comparison_op.__name__)(right, axis=1)

    @pytest.mark.parametrize('values, op', [([False, False, True, False], 'eq'), ([True, True, False, True], 'ne'), ([False, False, True, False], 'le'), ([False, False, False, False], 'lt'), ([False, True, True, False], 'ge'), ([False, True, False, False], 'gt')])
    def test_comparison_flex_alignment(self, values, op):
        left = Series([1, 3, 2], index=list('abc'))
        right = Series([2, 2, 2], index=list('bcd'))
        result = getattr(left, op)(right)
        expected = Series(values, index=list('abcd'))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values, op, fill_value', [([False, False, True, True], 'eq', 2), ([True, True, False, False], 'ne', 2), ([False, False, True, True], 'le', 0), ([False, False, False, True], 'lt', 0), ([True, True, True, False], 'ge', 0), ([True, True, False, False], 'gt', 0)])
    def test_comparison_flex_alignment_fill(self, values, op, fill_value):
        left = Series([1, 3, 2], index=list('abc'))
        right = Series([2, 2, 2], index=list('bcd'))
        result = getattr(left, op)(right, fill_value=fill_value)
        expected = Series(values, index=list('abcd'))
        tm.assert_series_equal(result, expected)