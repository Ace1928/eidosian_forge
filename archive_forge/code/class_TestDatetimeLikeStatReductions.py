import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDatetimeLikeStatReductions:

    @pytest.mark.parametrize('box', [Series, pd.Index, pd.array])
    def test_dt64_mean(self, tz_naive_fixture, box):
        tz = tz_naive_fixture
        dti = date_range('2001-01-01', periods=11, tz=tz)
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        dtarr = dti._data
        obj = box(dtarr)
        assert obj.mean() == pd.Timestamp('2001-01-06', tz=tz)
        assert obj.mean(skipna=False) == pd.Timestamp('2001-01-06', tz=tz)
        dtarr[-2] = pd.NaT
        obj = box(dtarr)
        assert obj.mean() == pd.Timestamp('2001-01-06 07:12:00', tz=tz)
        assert obj.mean(skipna=False) is pd.NaT

    @pytest.mark.parametrize('box', [Series, pd.Index, pd.array])
    @pytest.mark.parametrize('freq', ['s', 'h', 'D', 'W', 'B'])
    def test_period_mean(self, box, freq):
        dti = date_range('2001-01-01', periods=11)
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        warn = FutureWarning if freq == 'B' else None
        msg = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)
        obj = box(parr)
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean()
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean(skipna=True)
        parr[-2] = pd.NaT
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean()
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean(skipna=True)

    @pytest.mark.parametrize('box', [Series, pd.Index, pd.array])
    def test_td64_mean(self, box):
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], 'm8[D]')
        tdi = pd.TimedeltaIndex(m8values).as_unit('ns')
        tdarr = tdi._data
        obj = box(tdarr, copy=False)
        result = obj.mean()
        expected = np.array(tdarr).mean()
        assert result == expected
        tdarr[0] = pd.NaT
        assert obj.mean(skipna=False) is pd.NaT
        result2 = obj.mean(skipna=True)
        assert result2 == tdi[1:].mean()
        assert result2.round('us') == (result * 11.0 / 10).round('us')