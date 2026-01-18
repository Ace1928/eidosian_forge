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
def _check_series_roundtrip(s, type_=None, expected_pa_type=None):
    arr = pa.array(s, from_pandas=True, type=type_)
    if type_ is not None and expected_pa_type is None:
        expected_pa_type = type_
    if expected_pa_type is not None:
        assert arr.type == expected_pa_type
    result = pd.Series(arr.to_pandas(), name=s.name)
    tm.assert_series_equal(s, result)