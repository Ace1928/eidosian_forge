import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestSeriesPeriodValuesDtAccessor:

    @pytest.mark.parametrize('input_vals', [[Period('2016-01', freq='M'), Period('2016-02', freq='M')], [Period('2016-01-01', freq='D'), Period('2016-01-02', freq='D')], [Period('2016-01-01 00:00:00', freq='h'), Period('2016-01-01 01:00:00', freq='h')], [Period('2016-01-01 00:00:00', freq='M'), Period('2016-01-01 00:01:00', freq='M')], [Period('2016-01-01 00:00:00', freq='s'), Period('2016-01-01 00:00:01', freq='s')]])
    def test_end_time_timevalues(self, input_vals):
        input_vals = PeriodArray._from_sequence(np.asarray(input_vals))
        ser = Series(input_vals)
        result = ser.dt.end_time
        expected = ser.apply(lambda x: x.end_time)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input_vals', ['2001', 'NaT'])
    def test_to_period(self, input_vals):
        expected = Series([input_vals], dtype='Period[D]')
        result = Series([input_vals], dtype='datetime64[ns]').dt.to_period('D')
        tm.assert_series_equal(result, expected)