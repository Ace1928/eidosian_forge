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
def _alltypes_example(size=100):
    return pd.DataFrame({'uint8': np.arange(size, dtype=np.uint8), 'uint16': np.arange(size, dtype=np.uint16), 'uint32': np.arange(size, dtype=np.uint32), 'uint64': np.arange(size, dtype=np.uint64), 'int8': np.arange(size, dtype=np.int16), 'int16': np.arange(size, dtype=np.int16), 'int32': np.arange(size, dtype=np.int32), 'int64': np.arange(size, dtype=np.int64), 'float32': np.arange(size, dtype=np.float32), 'float64': np.arange(size, dtype=np.float64), 'bool': np.random.randn(size) > 0, 'datetime[s]': np.arange('2016-01-01T00:00:00.001', size, dtype='datetime64[s]'), 'datetime[ms]': np.arange('2016-01-01T00:00:00.001', size, dtype='datetime64[ms]'), 'datetime[us]': np.arange('2016-01-01T00:00:00.001', size, dtype='datetime64[us]'), 'datetime[ns]': np.arange('2016-01-01T00:00:00.001', size, dtype='datetime64[ns]'), 'timedelta64[s]': np.arange(0, size, dtype='timedelta64[s]'), 'timedelta64[ms]': np.arange(0, size, dtype='timedelta64[ms]'), 'timedelta64[us]': np.arange(0, size, dtype='timedelta64[us]'), 'timedelta64[ns]': np.arange(0, size, dtype='timedelta64[ns]'), 'str': [str(x) for x in range(size)], 'str_with_nulls': [None] + [str(x) for x in range(size - 2)] + [None], 'empty_str': [''] * size})