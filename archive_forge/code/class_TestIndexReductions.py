from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
class TestIndexReductions:

    @pytest.mark.parametrize('start,stop,step', [(0, 400, 3), (500, 0, -6), (-10 ** 6, 10 ** 6, 4), (10 ** 6, -10 ** 6, -4), (0, 10, 20)])
    def test_max_min_range(self, start, stop, step):
        idx = RangeIndex(start, stop, step)
        expected = idx._values.max()
        result = idx.max()
        assert result == expected
        result2 = idx.max(skipna=False)
        assert result2 == expected
        expected = idx._values.min()
        result = idx.min()
        assert result == expected
        result2 = idx.min(skipna=False)
        assert result2 == expected
        idx = RangeIndex(start, stop, -step)
        assert isna(idx.max())
        assert isna(idx.min())

    def test_minmax_timedelta64(self):
        idx1 = TimedeltaIndex(['1 days', '2 days', '3 days'])
        assert idx1.is_monotonic_increasing
        idx2 = TimedeltaIndex(['1 days', np.nan, '3 days', 'NaT'])
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Timedelta('1 days')
            assert idx.max() == Timedelta('3 days')
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize('op', ['min', 'max'])
    def test_minmax_timedelta_empty_or_na(self, op):
        obj = TimedeltaIndex([])
        assert getattr(obj, op)() is NaT
        obj = TimedeltaIndex([NaT])
        assert getattr(obj, op)() is NaT
        obj = TimedeltaIndex([NaT, NaT, NaT])
        assert getattr(obj, op)() is NaT

    def test_numpy_minmax_timedelta64(self):
        td = timedelta_range('16815 days', '16820 days', freq='D')
        assert np.min(td) == Timedelta('16815 days')
        assert np.max(td) == Timedelta('16820 days')
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)
        assert np.argmin(td) == 0
        assert np.argmax(td) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)

    def test_timedelta_ops(self):
        s = Series([Timestamp('20130101') + timedelta(seconds=i * i) for i in range(10)])
        td = s.diff()
        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected
        result = td.to_frame().mean()
        assert result[0] == expected
        result = td.quantile(0.1)
        expected = Timedelta(np.timedelta64(2600, 'ms'))
        assert result == expected
        result = td.median()
        expected = to_timedelta('00:00:09')
        assert result == expected
        result = td.to_frame().median()
        assert result[0] == expected
        result = td.sum()
        expected = to_timedelta('00:01:21')
        assert result == expected
        result = td.to_frame().sum()
        assert result[0] == expected
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected
        result = td.to_frame().std()
        assert result[0] == expected
        s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07')])
        assert s.diff().median() == timedelta(days=4)
        s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07'), Timestamp('2015-02-15')])
        assert s.diff().median() == timedelta(days=6)

    @pytest.mark.parametrize('opname', ['skew', 'kurt', 'sem', 'prod', 'var'])
    def test_invalid_td64_reductions(self, opname):
        s = Series([Timestamp('20130101') + timedelta(seconds=i * i) for i in range(10)])
        td = s.diff()
        msg = '|'.join([f"reduction operation '{opname}' not allowed for this dtype", f'cannot perform {opname} with type timedelta64\\[ns\\]', f"does not support reduction '{opname}'"])
        with pytest.raises(TypeError, match=msg):
            getattr(td, opname)()
        with pytest.raises(TypeError, match=msg):
            getattr(td.to_frame(), opname)(numeric_only=False)

    def test_minmax_tz(self, tz_naive_fixture):
        tz = tz_naive_fixture
        idx1 = DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], tz=tz)
        assert idx1.is_monotonic_increasing
        idx2 = DatetimeIndex(['2011-01-01', NaT, '2011-01-03', '2011-01-02', NaT], tz=tz)
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp('2011-01-01', tz=tz)
            assert idx.max() == Timestamp('2011-01-03', tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize('op', ['min', 'max'])
    def test_minmax_nat_datetime64(self, op):
        obj = DatetimeIndex([])
        assert isna(getattr(obj, op)())
        obj = DatetimeIndex([NaT])
        assert isna(getattr(obj, op)())
        obj = DatetimeIndex([NaT, NaT, NaT])
        assert isna(getattr(obj, op)())

    def test_numpy_minmax_integer(self):
        idx = Index([1, 2, 3])
        expected = idx.values.max()
        result = np.max(idx)
        assert result == expected
        expected = idx.values.min()
        result = np.min(idx)
        assert result == expected
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)
        expected = idx.values.argmax()
        result = np.argmax(idx)
        assert result == expected
        expected = idx.values.argmin()
        result = np.argmin(idx)
        assert result == expected
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(idx, out=0)

    def test_numpy_minmax_range(self):
        idx = RangeIndex(0, 10, 3)
        result = np.max(idx)
        assert result == 9
        result = np.min(idx)
        assert result == 0
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

    def test_numpy_minmax_datetime64(self):
        dr = date_range(start='2016-01-15', end='2016-01-20')
        assert np.min(dr) == Timestamp('2016-01-15 00:00:00')
        assert np.max(dr) == Timestamp('2016-01-20 00:00:00')
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(dr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(dr, out=0)
        assert np.argmin(dr) == 0
        assert np.argmax(dr) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(dr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(dr, out=0)

    def test_minmax_period(self):
        idx1 = PeriodIndex([NaT, '2011-01-01', '2011-01-02', '2011-01-03'], freq='D')
        assert not idx1.is_monotonic_increasing
        assert idx1[1:].is_monotonic_increasing
        idx2 = PeriodIndex(['2011-01-01', NaT, '2011-01-03', '2011-01-02', NaT], freq='D')
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Period('2011-01-01', freq='D')
            assert idx.max() == Period('2011-01-03', freq='D')
        assert idx1.argmin() == 1
        assert idx2.argmin() == 0
        assert idx1.argmax() == 3
        assert idx2.argmax() == 2

    @pytest.mark.parametrize('op', ['min', 'max'])
    @pytest.mark.parametrize('data', [[], [NaT], [NaT, NaT, NaT]])
    def test_minmax_period_empty_nat(self, op, data):
        obj = PeriodIndex(data, freq='M')
        result = getattr(obj, op)()
        assert result is NaT

    def test_numpy_minmax_period(self):
        pr = period_range(start='2016-01-15', end='2016-01-20')
        assert np.min(pr) == Period('2016-01-15', freq='D')
        assert np.max(pr) == Period('2016-01-20', freq='D')
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(pr, out=0)
        assert np.argmin(pr) == 0
        assert np.argmax(pr) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(pr, out=0)

    def test_min_max_categorical(self):
        ci = pd.CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=False)
        msg = 'Categorical is not ordered for operation min\\nyou can use .as_ordered\\(\\) to change the Categorical to an ordered one\\n'
        with pytest.raises(TypeError, match=msg):
            ci.min()
        msg = 'Categorical is not ordered for operation max\\nyou can use .as_ordered\\(\\) to change the Categorical to an ordered one\\n'
        with pytest.raises(TypeError, match=msg):
            ci.max()
        ci = pd.CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=True)
        assert ci.min() == 'c'
        assert ci.max() == 'b'