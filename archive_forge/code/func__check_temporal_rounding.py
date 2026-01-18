from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def _check_temporal_rounding(ts, values, unit):
    unit_shorthand = {'nanosecond': 'ns', 'microsecond': 'us', 'millisecond': 'L', 'second': 's', 'minute': 'min', 'hour': 'H', 'day': 'D'}
    greater_unit = {'nanosecond': 'us', 'microsecond': 'ms', 'millisecond': 's', 'second': 'min', 'minute': 'H', 'hour': 'd'}
    ta = pa.array(ts)
    for value in values:
        frequency = str(value) + unit_shorthand[unit]
        options = pc.RoundTemporalOptions(value, unit)
        result = pc.ceil_temporal(ta, options=options).to_pandas()
        expected = ts.dt.ceil(frequency)
        np.testing.assert_array_equal(result, expected)
        result = pc.floor_temporal(ta, options=options).to_pandas()
        expected = ts.dt.floor(frequency)
        np.testing.assert_array_equal(result, expected)
        result = pc.round_temporal(ta, options=options).to_pandas()
        expected = ts.dt.round(frequency)
        np.testing.assert_array_equal(result, expected)
        if unit != 'day':
            options = pc.RoundTemporalOptions(value, unit, calendar_based_origin=True)
            origin = ts.dt.floor(greater_unit[unit])
            if ta.type.tz is None:
                result = pc.ceil_temporal(ta, options=options).to_pandas()
                expected = (ts - origin).dt.ceil(frequency) + origin
                np.testing.assert_array_equal(result, expected)
            result = pc.floor_temporal(ta, options=options).to_pandas()
            expected = (ts - origin).dt.floor(frequency) + origin
            np.testing.assert_array_equal(result, expected)
            result = pc.round_temporal(ta, options=options).to_pandas()
            expected = (ts - origin).dt.round(frequency) + origin
            np.testing.assert_array_equal(result, expected)
        if unit == 'day':
            result = pc.ceil_temporal(ta, multiple=value).to_pandas()
            expected = ts.dt.ceil(frequency)
            np.testing.assert_array_equal(result, expected)
            result = pc.floor_temporal(ta, multiple=value).to_pandas()
            expected = ts.dt.floor(frequency)
            np.testing.assert_array_equal(result, expected)
            result = pc.round_temporal(ta, multiple=value).to_pandas()
            expected = ts.dt.round(frequency)
            np.testing.assert_array_equal(result, expected)
    if ta.type.tz is None:
        options = pc.RoundTemporalOptions(value, unit, ceil_is_strictly_greater=True)
        result = pc.ceil_temporal(ta, options=options)
        expected = ts.dt.ceil(frequency)
        expected = np.where(expected == ts, expected + pd.Timedelta(value, unit_shorthand[unit]), expected)
        np.testing.assert_array_equal(result, expected)
    if unit == 'day':
        frequency = '1D'
        result = pc.ceil_temporal(ta).to_pandas()
        expected = ts.dt.ceil(frequency)
        np.testing.assert_array_equal(result, expected)
        result = pc.floor_temporal(ta).to_pandas()
        expected = ts.dt.floor(frequency)
        np.testing.assert_array_equal(result, expected)
        result = pc.round_temporal(ta).to_pandas()
        expected = ts.dt.round(frequency)
        np.testing.assert_array_equal(result, expected)