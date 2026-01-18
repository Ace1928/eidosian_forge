from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def _check_cast_case(case, *, safe=True, check_array_construction=True):
    in_data, in_type, out_data, out_type = case
    if isinstance(out_data, pa.Array):
        assert out_data.type == out_type
        expected = out_data
    else:
        expected = pa.array(out_data, type=out_type)
    if isinstance(in_data, pa.Array):
        assert in_data.type == in_type
        in_arr = in_data
    else:
        in_arr = pa.array(in_data, type=in_type)
    casted = in_arr.cast(out_type, safe=safe)
    casted.validate(full=True)
    assert casted.equals(expected)
    if check_array_construction:
        in_arr = pa.array(in_data, type=out_type, safe=safe)
        assert in_arr.equals(expected)