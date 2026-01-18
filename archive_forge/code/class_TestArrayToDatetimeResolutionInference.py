from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
class TestArrayToDatetimeResolutionInference:

    def test_infer_all_nat(self):
        arr = np.array([NaT, np.nan], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        assert result.dtype == 'M8[s]'

    def test_infer_homogeoneous_datetimes(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        arr = np.array([dt, dt, dt], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, dt, dt], dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_date_objects(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt2 = dt.date()
        arr = np.array([None, dt2, dt2, dt2], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT'), dt2, dt2, dt2], dtype='M8[s]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_dt64(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt64 = np.datetime64(dt, 'ms')
        arr = np.array([None, dt64, dt64, dt64], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT'), dt64, dt64, dt64], dtype='M8[ms]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_timestamps(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        ts = Timestamp(dt).as_unit('ns')
        arr = np.array([None, ts, ts, ts], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT')] + [ts.asm8] * 3, dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_datetimes_strings(self):
        item = '2023-10-27 18:03:05.678000'
        arr = np.array([None, item, item, item], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64('NaT'), item, item, item], dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_heterogeneous(self):
        dtstr = '2023-10-27 18:03:05.678000'
        arr = np.array([dtstr, dtstr[:-3], dtstr[:-7], None], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(arr, dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)
        result, tz = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz is None
        tm.assert_numpy_array_equal(result, expected[::-1])

    @pytest.mark.parametrize('item', [float('nan'), NaT.value, float(NaT.value), 'NaT', ''])
    def test_infer_with_nat_int_float_str(self, item):
        dt = datetime(2023, 11, 15, 15, 5, 6)
        arr = np.array([dt, item], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, np.datetime64('NaT')], dtype='M8[us]')
        tm.assert_numpy_array_equal(result, expected)
        result2, tz2 = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz2 is None
        tm.assert_numpy_array_equal(result2, expected[::-1])