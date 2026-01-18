import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
class TestFromScalar:

    @pytest.fixture(params=[list, dict, None])
    def box(self, request):
        return request.param

    @pytest.fixture
    def constructor(self, frame_or_series, box):
        extra = {'index': range(2)}
        if frame_or_series is DataFrame:
            extra['columns'] = ['A']
        if box is None:
            return functools.partial(frame_or_series, **extra)
        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series({0: x, 1: x}, **extra, **kwargs)
            else:
                return lambda x, **kwargs: frame_or_series({'A': x}, **extra, **kwargs)
        elif frame_or_series is Series:
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
        else:
            return lambda x, **kwargs: frame_or_series({'A': [x, x]}, **extra, **kwargs)

    @pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
    def test_from_nat_scalar(self, dtype, constructor):
        obj = constructor(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    def test_from_timedelta_scalar_preserves_nanos(self, constructor):
        td = Timedelta(1)
        obj = constructor(td, dtype='m8[ns]')
        assert get1(obj) == td

    def test_from_timestamp_scalar_preserves_nanos(self, constructor, fixed_now_ts):
        ts = fixed_now_ts + Timedelta(1)
        obj = constructor(ts, dtype='M8[ns]')
        assert get1(obj) == ts

    def test_from_timedelta64_scalar_object(self, constructor):
        td = Timedelta(1)
        td64 = td.to_timedelta64()
        obj = constructor(td64, dtype=object)
        assert isinstance(get1(obj), np.timedelta64)

    @pytest.mark.parametrize('cls', [np.datetime64, np.timedelta64])
    def test_from_scalar_datetimelike_mismatched(self, constructor, cls):
        scalar = cls('NaT', 'ns')
        dtype = {np.datetime64: 'm8[ns]', np.timedelta64: 'M8[ns]'}[cls]
        if cls is np.datetime64:
            msg1 = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        else:
            msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
        msg = '|'.join(['Cannot cast', msg1])
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)
        scalar = cls(4, 'ns')
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

    @pytest.mark.parametrize('cls', [datetime, np.datetime64])
    def test_from_out_of_bounds_ns_datetime(self, constructor, cls, request, box, frame_or_series):
        if box is list or (frame_or_series is Series and box is dict):
            mark = pytest.mark.xfail(reason='Timestamp constructor has been updated to cast dt64 to non-nano, but DatetimeArray._from_sequence has not', strict=True)
            request.applymarker(mark)
        scalar = datetime(9999, 1, 1)
        exp_dtype = 'M8[us]'
        if cls is np.datetime64:
            scalar = np.datetime64(scalar, 'D')
            exp_dtype = 'M8[s]'
        result = constructor(scalar)
        item = get1(result)
        dtype = tm.get_dtype(result)
        assert type(item) is Timestamp
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.skip_ubsan
    def test_out_of_s_bounds_datetime64(self, constructor):
        scalar = np.datetime64(np.iinfo(np.int64).max, 'D')
        result = constructor(scalar)
        item = get1(result)
        assert type(item) is np.datetime64
        dtype = tm.get_dtype(result)
        assert dtype == object

    @pytest.mark.parametrize('cls', [timedelta, np.timedelta64])
    def test_from_out_of_bounds_ns_timedelta(self, constructor, cls, request, box, frame_or_series):
        if box is list or (frame_or_series is Series and box is dict):
            mark = pytest.mark.xfail(reason='TimedeltaArray constructor has been updated to cast td64 to non-nano, but TimedeltaArray._from_sequence has not', strict=True)
            request.applymarker(mark)
        scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
        exp_dtype = 'm8[us]'
        if cls is np.timedelta64:
            scalar = np.timedelta64(scalar, 'D')
            exp_dtype = 'm8[s]'
        result = constructor(scalar)
        item = get1(result)
        dtype = tm.get_dtype(result)
        assert type(item) is Timedelta
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.skip_ubsan
    @pytest.mark.parametrize('cls', [np.datetime64, np.timedelta64])
    def test_out_of_s_bounds_timedelta64(self, constructor, cls):
        scalar = cls(np.iinfo(np.int64).max, 'D')
        result = constructor(scalar)
        item = get1(result)
        assert type(item) is cls
        dtype = tm.get_dtype(result)
        assert dtype == object

    def test_tzaware_data_tznaive_dtype(self, constructor, box, frame_or_series):
        tz = 'US/Eastern'
        ts = Timestamp('2019', tz=tz)
        if box is None or (frame_or_series is DataFrame and box is dict):
            msg = 'Cannot unbox tzaware Timestamp to tznaive dtype'
            err = TypeError
        else:
            msg = 'Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series\\(values\\).dt.tz_localize\\(None\\) instead.'
            err = ValueError
        with pytest.raises(err, match=msg):
            constructor(ts, dtype='M8[ns]')