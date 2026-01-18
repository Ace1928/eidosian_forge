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
class TestNamedAggregationDataFrame:

    def test_agg_relabel(self):
        df = DataFrame({'group': ['a', 'a', 'b', 'b'], 'A': [0, 1, 2, 3], 'B': [5, 6, 7, 8]})
        result = df.groupby('group').agg(a_max=('A', 'max'), b_max=('B', 'max'))
        expected = DataFrame({'a_max': [1, 3], 'b_max': [6, 8]}, index=Index(['a', 'b'], name='group'), columns=['a_max', 'b_max'])
        tm.assert_frame_equal(result, expected)
        p98 = functools.partial(np.percentile, q=98)
        result = df.groupby('group').agg(b_min=('B', 'min'), a_min=('A', 'min'), a_mean=('A', 'mean'), a_max=('A', 'max'), b_max=('B', 'max'), a_98=('A', p98))
        expected = DataFrame({'b_min': [5, 7], 'a_min': [0, 2], 'a_mean': [0.5, 2.5], 'a_max': [1, 3], 'b_max': [6, 8], 'a_98': [0.98, 2.98]}, index=Index(['a', 'b'], name='group'), columns=['b_min', 'a_min', 'a_mean', 'a_max', 'b_max', 'a_98'])
        tm.assert_frame_equal(result, expected)

    def test_agg_relabel_non_identifier(self):
        df = DataFrame({'group': ['a', 'a', 'b', 'b'], 'A': [0, 1, 2, 3], 'B': [5, 6, 7, 8]})
        result = df.groupby('group').agg(**{'my col': ('A', 'max')})
        expected = DataFrame({'my col': [1, 3]}, index=Index(['a', 'b'], name='group'))
        tm.assert_frame_equal(result, expected)

    def test_duplicate_no_raises(self):
        df = DataFrame({'A': [0, 0, 1, 1], 'B': [1, 2, 3, 4]})
        grouped = df.groupby('A').agg(a=('B', 'min'), b=('B', 'min'))
        expected = DataFrame({'a': [1, 3], 'b': [1, 3]}, index=Index([0, 1], name='A'))
        tm.assert_frame_equal(grouped, expected)
        quant50 = functools.partial(np.percentile, q=50)
        quant70 = functools.partial(np.percentile, q=70)
        quant50.__name__ = 'quant50'
        quant70.__name__ = 'quant70'
        test = DataFrame({'col1': ['a', 'a', 'b', 'b', 'b'], 'col2': [1, 2, 3, 4, 5]})
        grouped = test.groupby('col1').agg(quantile_50=('col2', quant50), quantile_70=('col2', quant70))
        expected = DataFrame({'quantile_50': [1.5, 4.0], 'quantile_70': [1.7, 4.4]}, index=Index(['a', 'b'], name='col1'))
        tm.assert_frame_equal(grouped, expected)

    def test_agg_relabel_with_level(self):
        df = DataFrame({'A': [0, 0, 1, 1], 'B': [1, 2, 3, 4]}, index=MultiIndex.from_product([['A', 'B'], ['a', 'b']]))
        result = df.groupby(level=0).agg(aa=('A', 'max'), bb=('A', 'min'), cc=('B', 'mean'))
        expected = DataFrame({'aa': [0, 1], 'bb': [0, 1], 'cc': [1.5, 3.5]}, index=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    def test_agg_relabel_other_raises(self):
        df = DataFrame({'A': [0, 0, 1], 'B': [1, 2, 3]})
        grouped = df.groupby('A')
        match = 'Must provide'
        with pytest.raises(TypeError, match=match):
            grouped.agg(foo=1)
        with pytest.raises(TypeError, match=match):
            grouped.agg()
        with pytest.raises(TypeError, match=match):
            grouped.agg(a=('B', 'max'), b=(1, 2, 3))

    def test_missing_raises(self):
        df = DataFrame({'A': [0, 1], 'B': [1, 2]})
        match = re.escape("Column(s) ['C'] do not exist")
        with pytest.raises(KeyError, match=match):
            df.groupby('A').agg(c=('C', 'sum'))

    def test_agg_namedtuple(self):
        df = DataFrame({'A': [0, 1], 'B': [1, 2]})
        result = df.groupby('A').agg(b=pd.NamedAgg('B', 'sum'), c=pd.NamedAgg(column='B', aggfunc='count'))
        expected = df.groupby('A').agg(b=('B', 'sum'), c=('B', 'count'))
        tm.assert_frame_equal(result, expected)

    def test_mangled(self):
        df = DataFrame({'A': [0, 1], 'B': [1, 2], 'C': [3, 4]})
        result = df.groupby('A').agg(b=('B', lambda x: 0), c=('C', lambda x: 1))
        expected = DataFrame({'b': [0, 0], 'c': [1, 1]}, index=Index([0, 1], name='A'))
        tm.assert_frame_equal(result, expected)