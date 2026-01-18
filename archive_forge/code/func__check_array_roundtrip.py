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
def _check_array_roundtrip(values, expected=None, mask=None, type=None):
    arr = pa.array(values, from_pandas=True, mask=mask, type=type)
    result = arr.to_pandas()
    values_nulls = pd.isnull(values)
    if mask is None:
        assert arr.null_count == values_nulls.sum()
    else:
        assert arr.null_count == (mask | values_nulls).sum()
    if expected is None:
        if mask is None:
            expected = pd.Series(values)
        else:
            expected = pd.Series(values).copy()
            expected[mask.copy()] = None
    tm.assert_series_equal(pd.Series(result), expected, check_names=False)