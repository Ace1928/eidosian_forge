from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
class TestDatetimeIndexSetOps:
    tz = [None, 'UTC', 'Asia/Tokyo', 'US/Eastern', 'dateutil/Asia/Singapore', 'dateutil/US/Pacific']

    def test_union2(self, sort):
        everything = date_range('2020-01-01', periods=10)
        first = everything[:5]
        second = everything[5:]
        union = first.union(second, sort=sort)
        tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize('box', [np.array, Series, list])
    def test_union3(self, sort, box):
        everything = date_range('2020-01-01', periods=10)
        first = everything[:5]
        second = everything[5:]
        expected = first.union(second, sort=sort)
        case = box(second.values)
        result = first.union(case, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', tz)
    def test_union(self, tz, sort):
        rng1 = date_range('1/1/2000', freq='D', periods=5, tz=tz)
        other1 = date_range('1/6/2000', freq='D', periods=5, tz=tz)
        expected1 = date_range('1/1/2000', freq='D', periods=10, tz=tz)
        expected1_notsorted = DatetimeIndex(list(other1) + list(rng1))
        rng2 = date_range('1/1/2000', freq='D', periods=5, tz=tz)
        other2 = date_range('1/4/2000', freq='D', periods=5, tz=tz)
        expected2 = date_range('1/1/2000', freq='D', periods=8, tz=tz)
        expected2_notsorted = DatetimeIndex(list(other2) + list(rng2[:3]))
        rng3 = date_range('1/1/2000', freq='D', periods=5, tz=tz)
        other3 = DatetimeIndex([], tz=tz).as_unit('ns')
        expected3 = date_range('1/1/2000', freq='D', periods=5, tz=tz)
        expected3_notsorted = rng3
        for rng, other, exp, exp_notsorted in [(rng1, other1, expected1, expected1_notsorted), (rng2, other2, expected2, expected2_notsorted), (rng3, other3, expected3, expected3_notsorted)]:
            result_union = rng.union(other, sort=sort)
            tm.assert_index_equal(result_union, exp)
            result_union = other.union(rng, sort=sort)
            if sort is None:
                tm.assert_index_equal(result_union, exp)
            else:
                tm.assert_index_equal(result_union, exp_notsorted)

    def test_union_coverage(self, sort):
        idx = DatetimeIndex(['2000-01-03', '2000-01-01', '2000-01-02'])
        ordered = DatetimeIndex(idx.sort_values(), freq='infer')
        result = ordered.union(idx, sort=sort)
        tm.assert_index_equal(result, ordered)
        result = ordered[:0].union(ordered, sort=sort)
        tm.assert_index_equal(result, ordered)
        assert result.freq == ordered.freq

    def test_union_bug_1730(self, sort):
        rng_a = date_range('1/1/2012', periods=4, freq='3h')
        rng_b = date_range('1/1/2012', periods=4, freq='4h')
        result = rng_a.union(rng_b, sort=sort)
        exp = list(rng_a) + list(rng_b[1:])
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_bug_1745(self, sort):
        left = DatetimeIndex(['2012-05-11 15:19:49.695000'])
        right = DatetimeIndex(['2012-05-29 13:04:21.322000', '2012-05-11 15:27:24.873000', '2012-05-11 15:31:05.350000'])
        result = left.union(right, sort=sort)
        exp = DatetimeIndex(['2012-05-11 15:19:49.695000', '2012-05-29 13:04:21.322000', '2012-05-11 15:27:24.873000', '2012-05-11 15:31:05.350000'])
        if sort is None:
            exp = exp.sort_values()
        tm.assert_index_equal(result, exp)

    def test_union_bug_4564(self, sort):
        from pandas import DateOffset
        left = date_range('2013-01-01', '2013-02-01')
        right = left + DateOffset(minutes=15)
        result = left.union(right, sort=sort)
        exp = list(left) + list(right)
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_freq_both_none(self, sort):
        expected = bdate_range('20150101', periods=10)
        expected._data.freq = None
        result = expected.union(expected, sort=sort)
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_union_freq_infer(self):
        dti = date_range('2016-01-01', periods=5)
        left = dti[[0, 1, 3, 4]]
        right = dti[[2, 3, 1]]
        assert left.freq is None
        assert right.freq is None
        result = left.union(right)
        tm.assert_index_equal(result, dti)
        assert result.freq == 'D'

    def test_union_dataframe_index(self):
        rng1 = date_range('1/1/1999', '1/1/2012', freq='MS')
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)
        rng2 = date_range('1/1/1980', '12/1/2001', freq='MS')
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({'s1': s1, 's2': s2})
        exp = date_range('1/1/1980', '1/1/2012', freq='MS')
        tm.assert_index_equal(df.index, exp)

    def test_union_with_DatetimeIndex(self, sort):
        i1 = Index(np.arange(0, 20, 2, dtype=np.int64))
        i2 = date_range(start='2012-01-03 00:00:00', periods=10, freq='D')
        i1.union(i2, sort=sort)
        i2.union(i1, sort=sort)

    def test_union_same_timezone_different_units(self):
        idx1 = date_range('2000-01-01', periods=3, tz='UTC').as_unit('ms')
        idx2 = date_range('2000-01-01', periods=3, tz='UTC').as_unit('us')
        result = idx1.union(idx2)
        expected = date_range('2000-01-01', periods=3, tz='UTC').as_unit('us')
        tm.assert_index_equal(result, expected)

    def test_intersection2(self):
        first = date_range('2020-01-01', periods=10)
        second = first[5:]
        intersect = first.intersection(second)
        tm.assert_index_equal(intersect, second)
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.intersection(case)
            tm.assert_index_equal(result, second)
        third = Index(['a', 'b', 'c'])
        result = first.intersection(third)
        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo', 'US/Eastern', 'dateutil/US/Pacific'])
    def test_intersection(self, tz, sort):
        base = date_range('6/1/2000', '6/30/2000', freq='D', name='idx')
        rng2 = date_range('5/15/2000', '6/20/2000', freq='D', name='idx')
        expected2 = date_range('6/1/2000', '6/20/2000', freq='D', name='idx')
        rng3 = date_range('5/15/2000', '6/20/2000', freq='D', name='other')
        expected3 = date_range('6/1/2000', '6/20/2000', freq='D', name=None)
        rng4 = date_range('7/1/2000', '7/31/2000', freq='D', name='idx')
        expected4 = DatetimeIndex([], freq='D', name='idx', dtype='M8[ns]')
        for rng, expected in [(rng2, expected2), (rng3, expected3), (rng4, expected4)]:
            result = base.intersection(rng)
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
        base = DatetimeIndex(['2011-01-05', '2011-01-04', '2011-01-02', '2011-01-03'], tz=tz, name='idx').as_unit('ns')
        rng2 = DatetimeIndex(['2011-01-04', '2011-01-02', '2011-02-02', '2011-02-03'], tz=tz, name='idx').as_unit('ns')
        expected2 = DatetimeIndex(['2011-01-04', '2011-01-02'], tz=tz, name='idx').as_unit('ns')
        rng3 = DatetimeIndex(['2011-01-04', '2011-01-02', '2011-02-02', '2011-02-03'], tz=tz, name='other').as_unit('ns')
        expected3 = DatetimeIndex(['2011-01-04', '2011-01-02'], tz=tz, name=None).as_unit('ns')
        rng4 = date_range('7/1/2000', '7/31/2000', freq='D', tz=tz, name='idx')
        expected4 = DatetimeIndex([], tz=tz, name='idx').as_unit('ns')
        assert expected4.freq is None
        for rng, expected in [(rng2, expected2), (rng3, expected3), (rng4, expected4)]:
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

    @pytest.mark.parametrize('freq', ['min', 'B'])
    def test_intersection_empty(self, tz_aware_fixture, freq):
        tz = tz_aware_fixture
        rng = date_range('6/1/2000', '6/15/2000', freq=freq, tz=tz)
        result = rng[0:0].intersection(rng)
        assert len(result) == 0
        assert result.freq == rng.freq
        result = rng.intersection(rng[0:0])
        assert len(result) == 0
        assert result.freq == rng.freq
        check_freq = freq != 'min'
        result = rng[:3].intersection(rng[-3:])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            assert result.freq == rng.freq
        result = rng[-3:].intersection(rng[:3])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            assert result.freq == rng.freq

    def test_intersection_bug_1708(self):
        from pandas import DateOffset
        index_1 = date_range('1/1/2012', periods=4, freq='12h')
        index_2 = index_1 + DateOffset(hours=1)
        result = index_1.intersection(index_2)
        assert len(result) == 0

    @pytest.mark.parametrize('tz', tz)
    def test_difference(self, tz, sort):
        rng_dates = ['1/2/2000', '1/3/2000', '1/1/2000', '1/4/2000', '1/5/2000']
        rng1 = DatetimeIndex(rng_dates, tz=tz)
        other1 = date_range('1/6/2000', freq='D', periods=5, tz=tz)
        expected1 = DatetimeIndex(rng_dates, tz=tz)
        rng2 = DatetimeIndex(rng_dates, tz=tz)
        other2 = date_range('1/4/2000', freq='D', periods=5, tz=tz)
        expected2 = DatetimeIndex(rng_dates[:3], tz=tz)
        rng3 = DatetimeIndex(rng_dates, tz=tz)
        other3 = DatetimeIndex([], tz=tz)
        expected3 = DatetimeIndex(rng_dates, tz=tz)
        for rng, other, expected in [(rng1, other1, expected1), (rng2, other2, expected2), (rng3, other3, expected3)]:
            result_diff = rng.difference(other, sort)
            if sort is None and len(other):
                expected = expected.sort_values()
            tm.assert_index_equal(result_diff, expected)

    def test_difference_freq(self, sort):
        index = date_range('20160920', '20160925', freq='D')
        other = date_range('20160921', '20160924', freq='D')
        expected = DatetimeIndex(['20160920', '20160925'], dtype='M8[ns]', freq=None)
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal('freq', idx_diff, expected)
        other = date_range('20160922', '20160925', freq='D')
        idx_diff = index.difference(other, sort)
        expected = DatetimeIndex(['20160920', '20160921'], dtype='M8[ns]', freq='D')
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal('freq', idx_diff, expected)

    def test_datetimeindex_diff(self, sort):
        dti1 = date_range(freq='QE-JAN', start=datetime(1997, 12, 31), periods=100)
        dti2 = date_range(freq='QE-JAN', start=datetime(1997, 12, 31), periods=98)
        assert len(dti1.difference(dti2, sort)) == 2

    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo', 'US/Eastern'])
    def test_setops_preserve_freq(self, tz):
        rng = date_range('1/1/2000', '1/1/2002', name='idx', tz=tz)
        result = rng[:50].union(rng[50:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz
        result = rng[:50].union(rng[30:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz
        result = rng[:50].union(rng[60:100])
        assert result.name == rng.name
        assert result.freq is None
        assert result.tz == rng.tz
        result = rng[:50].intersection(rng[25:75])
        assert result.name == rng.name
        assert result.freqstr == 'D'
        assert result.tz == rng.tz
        nofreq = DatetimeIndex(list(rng[25:75]), name='other')
        result = rng[:50].union(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz
        result = rng[:50].intersection(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

    def test_intersection_non_tick_no_fastpath(self):
        dti = DatetimeIndex(['2018-12-31', '2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31', '2020-03-31'], freq='QE-DEC')
        result = dti[::2].intersection(dti[1::2])
        expected = dti[:0]
        tm.assert_index_equal(result, expected)

    def test_dti_intersection(self):
        rng = date_range('1/1/2011', periods=100, freq='h', tz='utc')
        left = rng[10:90][::-1]
        right = rng[20:80][::-1]
        assert left.tz == rng.tz
        result = left.intersection(right)
        assert result.tz == left.tz

    @pytest.mark.parametrize('setop', ['union', 'intersection', 'symmetric_difference'])
    def test_dti_setop_aware(self, setop):
        rng = date_range('2012-11-15 00:00:00', periods=6, freq='h', tz='US/Central')
        rng2 = date_range('2012-11-15 12:00:00', periods=6, freq='h', tz='US/Eastern')
        result = getattr(rng, setop)(rng2)
        left = rng.tz_convert('UTC')
        right = rng2.tz_convert('UTC')
        expected = getattr(left, setop)(right)
        tm.assert_index_equal(result, expected)
        assert result.tz == left.tz
        if len(result):
            assert result[0].tz is timezone.utc
            assert result[-1].tz is timezone.utc

    def test_dti_union_mixed(self):
        rng = DatetimeIndex([Timestamp('2011-01-01'), pd.NaT])
        rng2 = DatetimeIndex(['2012-01-01', '2012-01-02'], tz='Asia/Tokyo')
        result = rng.union(rng2)
        expected = Index([Timestamp('2011-01-01'), pd.NaT, Timestamp('2012-01-01', tz='Asia/Tokyo'), Timestamp('2012-01-02', tz='Asia/Tokyo')], dtype=object)
        tm.assert_index_equal(result, expected)