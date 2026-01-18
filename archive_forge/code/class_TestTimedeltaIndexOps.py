from pandas import (
import pandas._testing as tm
class TestTimedeltaIndexOps:

    def test_infer_freq(self, freq_sample):
        idx = timedelta_range('1', freq=freq_sample, periods=10)
        result = TimedeltaIndex(idx.asi8, freq='infer')
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample