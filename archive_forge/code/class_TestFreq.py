import pytest
from pandas import (
from pandas.tseries.offsets import (
class TestFreq:

    def test_freq_setter_errors(self):
        idx = DatetimeIndex(['20180101', '20180103', '20180105'])
        msg = 'Inferred frequency 2D from passed values does not conform to passed frequency 5D'
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = '5D'
        with pytest.raises(ValueError, match='Invalid frequency'):
            idx._data.freq = 'foo'

    @pytest.mark.parametrize('values', [['20180101', '20180103', '20180105'], []])
    @pytest.mark.parametrize('freq', ['2D', Day(2), '2B', BDay(2), '48h', Hour(48)])
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_freq_setter(self, values, freq, tz):
        idx = DatetimeIndex(values, tz=tz)
        idx._data.freq = freq
        assert idx.freq == freq
        assert isinstance(idx.freq, DateOffset)
        idx._data.freq = None
        assert idx.freq is None

    def test_freq_view_safe(self):
        dti = date_range('2016-01-01', periods=5)
        dta = dti._data
        dti2 = DatetimeIndex(dta)._with_freq(None)
        assert dti2.freq is None
        assert dti.freq == 'D'
        assert dta.freq == 'D'