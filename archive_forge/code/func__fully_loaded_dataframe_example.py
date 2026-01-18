import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def _fully_loaded_dataframe_example():
    index = pd.MultiIndex.from_arrays([pd.date_range('2000-01-01', periods=5).repeat(2), np.tile(np.array(['foo', 'bar'], dtype=object), 5)])
    c1 = pd.date_range('2000-01-01', periods=10)
    data = {0: c1, 1: c1.tz_localize('utc'), 2: c1.tz_localize('US/Eastern'), 3: c1[::2].tz_localize('utc').repeat(2).astype('category'), 4: ['foo', 'bar'] * 5, 5: pd.Series(['foo', 'bar'] * 5).astype('category').values, 6: [True, False] * 5, 7: np.random.randn(10), 8: np.random.randint(0, 100, size=10), 9: pd.period_range('2013', periods=10, freq='M'), 10: pd.interval_range(start=1, freq=1, periods=10)}
    return pd.DataFrame(data, index=index)