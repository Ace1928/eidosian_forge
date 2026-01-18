from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
class TestGetGroup:

    def test_get_group(self):
        df = DataFrame({'DATE': pd.to_datetime(['10-Oct-2013', '10-Oct-2013', '10-Oct-2013', '11-Oct-2013', '11-Oct-2013', '11-Oct-2013']), 'label': ['foo', 'foo', 'bar', 'foo', 'foo', 'bar'], 'VAL': [1, 2, 3, 4, 5, 6]})
        g = df.groupby('DATE')
        key = next(iter(g.groups))
        result1 = g.get_group(key)
        result2 = g.get_group(Timestamp(key).to_pydatetime())
        result3 = g.get_group(str(Timestamp(key)))
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        g = df.groupby(['DATE', 'label'])
        key = next(iter(g.groups))
        result1 = g.get_group(key)
        result2 = g.get_group((Timestamp(key[0]).to_pydatetime(), key[1]))
        result3 = g.get_group((str(Timestamp(key[0])), key[1]))
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        msg = 'must supply a tuple to get_group with multiple grouping keys'
        with pytest.raises(ValueError, match=msg):
            g.get_group('foo')
        with pytest.raises(ValueError, match=msg):
            g.get_group('foo')
        msg = 'must supply a same-length tuple to get_group with multiple grouping keys'
        with pytest.raises(ValueError, match=msg):
            g.get_group(('foo', 'bar', 'baz'))

    def test_get_group_empty_bins(self, observed):
        d = DataFrame([3, 1, 7, 6])
        bins = [0, 5, 10, 15]
        g = d.groupby(pd.cut(d[0], bins), observed=observed)
        result = g.get_group(pd.Interval(0, 5))
        expected = DataFrame([3, 1], index=[0, 1])
        tm.assert_frame_equal(result, expected)
        msg = "Interval\\(10, 15, closed='right'\\)"
        with pytest.raises(KeyError, match=msg):
            g.get_group(pd.Interval(10, 15))

    def test_get_group_grouped_by_tuple(self):
        df = DataFrame([[(1,), (1, 2), (1,), (1, 2)]], index=['ids']).T
        gr = df.groupby('ids')
        expected = DataFrame({'ids': [(1,), (1,)]}, index=[0, 2])
        result = gr.get_group((1,))
        tm.assert_frame_equal(result, expected)
        dt = pd.to_datetime(['2010-01-01', '2010-01-02', '2010-01-01', '2010-01-02'])
        df = DataFrame({'ids': [(x,) for x in dt]})
        gr = df.groupby('ids')
        result = gr.get_group(('2010-01-01',))
        expected = DataFrame({'ids': [(dt[0],), (dt[0],)]}, index=[0, 2])
        tm.assert_frame_equal(result, expected)

    def test_get_group_grouped_by_tuple_with_lambda(self):
        df = DataFrame({'Tuples': ((x, y) for x in [0, 1] for y in np.random.default_rng(2).integers(3, 5, 5))})
        gb = df.groupby('Tuples')
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])
        expected = gb.get_group(next(iter(gb.groups.keys())))
        result = gb_lambda.get_group(next(iter(gb_lambda.groups.keys())))
        tm.assert_frame_equal(result, expected)

    def test_groupby_with_empty(self):
        index = pd.DatetimeIndex(())
        data = ()
        series = Series(data, index, dtype=object)
        grouper = Grouper(freq='D')
        grouped = series.groupby(grouper)
        assert next(iter(grouped), None) is None

    def test_groupby_with_single_column(self):
        df = DataFrame({'a': list('abssbab')})
        tm.assert_frame_equal(df.groupby('a').get_group('a'), df.iloc[[0, 5]])
        exp = DataFrame(index=Index(['a', 'b', 's'], name='a'), columns=[])
        tm.assert_frame_equal(df.groupby('a').count(), exp)
        tm.assert_frame_equal(df.groupby('a').sum(), exp)
        exp = df.iloc[[3, 4, 5]]
        tm.assert_frame_equal(df.groupby('a').nth(1), exp)

    def test_gb_key_len_equal_axis_len(self):
        df = DataFrame([['foo', 'bar', 'B', 1], ['foo', 'bar', 'B', 2], ['foo', 'baz', 'C', 3]], columns=['first', 'second', 'third', 'one'])
        df = df.set_index(['first', 'second'])
        df = df.groupby(['first', 'second', 'third']).size()
        assert df.loc['foo', 'bar', 'B'] == 2
        assert df.loc['foo', 'baz', 'C'] == 1