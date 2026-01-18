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
def _parse_iso8601_without_reso(date_type, datetime_str):
    date, _ = _parse_iso8601_with_reso(date_type, datetime_str)
    return date