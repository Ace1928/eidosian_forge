import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestNumberScalar:

    def test_is_number(self):
        assert is_number(True)
        assert is_number(1)
        assert is_number(1.1)
        assert is_number(1 + 3j)
        assert is_number(np.int64(1))
        assert is_number(np.float64(1.1))
        assert is_number(np.complex128(1 + 3j))
        assert is_number(np.nan)
        assert not is_number(None)
        assert not is_number('x')
        assert not is_number(datetime(2011, 1, 1))
        assert not is_number(np.datetime64('2011-01-01'))
        assert not is_number(Timestamp('2011-01-01'))
        assert not is_number(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_number(timedelta(1000))
        assert not is_number(Timedelta('1 days'))
        assert not is_number(np.bool_(False))
        assert is_number(np.timedelta64(1, 'D'))

    def test_is_bool(self):
        assert is_bool(True)
        assert is_bool(False)
        assert is_bool(np.bool_(False))
        assert not is_bool(1)
        assert not is_bool(1.1)
        assert not is_bool(1 + 3j)
        assert not is_bool(np.int64(1))
        assert not is_bool(np.float64(1.1))
        assert not is_bool(np.complex128(1 + 3j))
        assert not is_bool(np.nan)
        assert not is_bool(None)
        assert not is_bool('x')
        assert not is_bool(datetime(2011, 1, 1))
        assert not is_bool(np.datetime64('2011-01-01'))
        assert not is_bool(Timestamp('2011-01-01'))
        assert not is_bool(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_bool(timedelta(1000))
        assert not is_bool(np.timedelta64(1, 'D'))
        assert not is_bool(Timedelta('1 days'))

    def test_is_integer(self):
        assert is_integer(1)
        assert is_integer(np.int64(1))
        assert not is_integer(True)
        assert not is_integer(1.1)
        assert not is_integer(1 + 3j)
        assert not is_integer(False)
        assert not is_integer(np.bool_(False))
        assert not is_integer(np.float64(1.1))
        assert not is_integer(np.complex128(1 + 3j))
        assert not is_integer(np.nan)
        assert not is_integer(None)
        assert not is_integer('x')
        assert not is_integer(datetime(2011, 1, 1))
        assert not is_integer(np.datetime64('2011-01-01'))
        assert not is_integer(Timestamp('2011-01-01'))
        assert not is_integer(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_integer(timedelta(1000))
        assert not is_integer(Timedelta('1 days'))
        assert not is_integer(np.timedelta64(1, 'D'))

    def test_is_float(self):
        assert is_float(1.1)
        assert is_float(np.float64(1.1))
        assert is_float(np.nan)
        assert not is_float(True)
        assert not is_float(1)
        assert not is_float(1 + 3j)
        assert not is_float(False)
        assert not is_float(np.bool_(False))
        assert not is_float(np.int64(1))
        assert not is_float(np.complex128(1 + 3j))
        assert not is_float(None)
        assert not is_float('x')
        assert not is_float(datetime(2011, 1, 1))
        assert not is_float(np.datetime64('2011-01-01'))
        assert not is_float(Timestamp('2011-01-01'))
        assert not is_float(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_float(timedelta(1000))
        assert not is_float(np.timedelta64(1, 'D'))
        assert not is_float(Timedelta('1 days'))

    def test_is_datetime_dtypes(self):
        ts = pd.date_range('20130101', periods=3)
        tsa = pd.date_range('20130101', periods=3, tz='US/Eastern')
        msg = 'is_datetime64tz_dtype is deprecated'
        assert is_datetime64_dtype('datetime64')
        assert is_datetime64_dtype('datetime64[ns]')
        assert is_datetime64_dtype(ts)
        assert not is_datetime64_dtype(tsa)
        assert not is_datetime64_ns_dtype('datetime64')
        assert is_datetime64_ns_dtype('datetime64[ns]')
        assert is_datetime64_ns_dtype(ts)
        assert is_datetime64_ns_dtype(tsa)
        assert is_datetime64_any_dtype('datetime64')
        assert is_datetime64_any_dtype('datetime64[ns]')
        assert is_datetime64_any_dtype(ts)
        assert is_datetime64_any_dtype(tsa)
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert not is_datetime64tz_dtype('datetime64')
            assert not is_datetime64tz_dtype('datetime64[ns]')
            assert not is_datetime64tz_dtype(ts)
            assert is_datetime64tz_dtype(tsa)

    @pytest.mark.parametrize('tz', ['US/Eastern', 'UTC'])
    def test_is_datetime_dtypes_with_tz(self, tz):
        dtype = f'datetime64[ns, {tz}]'
        assert not is_datetime64_dtype(dtype)
        msg = 'is_datetime64tz_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_any_dtype(dtype)

    def test_is_timedelta(self):
        assert is_timedelta64_dtype('timedelta64')
        assert is_timedelta64_dtype('timedelta64[ns]')
        assert not is_timedelta64_ns_dtype('timedelta64')
        assert is_timedelta64_ns_dtype('timedelta64[ns]')
        tdi = TimedeltaIndex([100000000000000.0, 200000000000000.0], dtype='timedelta64[ns]')
        assert is_timedelta64_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi.astype('timedelta64[ns]'))
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.float64))
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.int64))