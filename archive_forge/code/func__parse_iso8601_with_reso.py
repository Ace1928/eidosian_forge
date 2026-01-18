from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def _parse_iso8601_with_reso(date_type, timestr):
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    default = date_type(1, 1, 1)
    result = parse_iso8601_like(timestr)
    replace = {}
    for attr in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        value = result.get(attr, None)
        if value is not None:
            replace[attr] = int(value)
            resolution = attr
    return (default.replace(**replace), resolution)