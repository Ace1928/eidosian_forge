import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
class TestNamedAggregationSeries:

    def test_series_named_agg(self):
        df = Series([1, 2, 3, 4])
        gr = df.groupby([0, 0, 1, 1])
        result = gr.agg(a='sum', b='min')
        expected = DataFrame({'a': [3, 7], 'b': [1, 3]}, columns=['a', 'b'], index=np.array([0, 1]))
        tm.assert_frame_equal(result, expected)
        result = gr.agg(b='min', a='sum')
        expected = expected[['b', 'a']]
        tm.assert_frame_equal(result, expected)

    def test_no_args_raises(self):
        gr = Series([1, 2]).groupby([0, 1])
        with pytest.raises(TypeError, match='Must provide'):
            gr.agg()
        result = gr.agg([])
        expected = DataFrame(columns=[])
        tm.assert_frame_equal(result, expected)

    def test_series_named_agg_duplicates_no_raises(self):
        gr = Series([1, 2, 3]).groupby([0, 0, 1])
        grouped = gr.agg(a='sum', b='sum')
        expected = DataFrame({'a': [3, 3], 'b': [3, 3]}, index=np.array([0, 1]))
        tm.assert_frame_equal(expected, grouped)

    def test_mangled(self):
        gr = Series([1, 2, 3]).groupby([0, 0, 1])
        result = gr.agg(a=lambda x: 0, b=lambda x: 1)
        expected = DataFrame({'a': [0, 0], 'b': [1, 1]}, index=np.array([0, 1]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('inp', [pd.NamedAgg(column='anything', aggfunc='min'), ('anything', 'min'), ['anything', 'min']])
    def test_named_agg_nametuple(self, inp):
        s = Series([1, 1, 2, 2, 3, 3, 4, 5])
        msg = f'func is expected but received {type(inp).__name__}'
        with pytest.raises(TypeError, match=msg):
            s.groupby(s.values).agg(a=inp)