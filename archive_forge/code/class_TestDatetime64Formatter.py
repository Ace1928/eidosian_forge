from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
class TestDatetime64Formatter:

    def test_mixed(self):
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 1, 12), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == '2013-01-01 00:00:00'
        assert result[1].strip() == '2013-01-01 12:00:00'

    def test_dates(self):
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 2), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == '2013-01-01'
        assert result[1].strip() == '2013-01-02'

    def test_date_nanos(self):
        x = Series([Timestamp(200)])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == '1970-01-01 00:00:00.000000200'

    def test_dates_display(self):
        x = Series(date_range('20130101 09:00:00', periods=5, freq='D'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-05 09:00:00'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='s'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:04'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='ms'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00.000'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:00.004'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='us'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00.000000'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:00.000004'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='ns'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00.000000000'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:00.000000004'

    def test_datetime64formatter_yearmonth(self):
        x = Series([datetime(2016, 1, 1), datetime(2016, 2, 2)])._values

        def format_func(x):
            return x.strftime('%Y-%m')
        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        result = formatter.get_result()
        assert result == ['2016-01', '2016-02']

    def test_datetime64formatter_hoursecond(self):
        x = Series(pd.to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f'))._values

        def format_func(x):
            return x.strftime('%H:%M')
        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        result = formatter.get_result()
        assert result == ['10:10', '12:12']

    def test_datetime64formatter_tz_ms(self):
        x = Series(np.array(['2999-01-01', '2999-01-02', 'NaT'], dtype='datetime64[ms]')).dt.tz_localize('US/Pacific')._values
        result = fmt._Datetime64TZFormatter(x).get_result()
        assert result[0].strip() == '2999-01-01 00:00:00-08:00'
        assert result[1].strip() == '2999-01-02 00:00:00-08:00'