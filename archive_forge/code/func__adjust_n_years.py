from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import (
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import (
from xarray.core.utils import emit_user_level_warning
def _adjust_n_years(other, n, month, reference_day):
    """Adjust the number of times an annual offset is applied based on
    another date, and the reference day provided"""
    if n > 0:
        if other.month < month or (other.month == month and other.day < reference_day):
            n -= 1
    elif other.month > month or (other.month == month and other.day > reference_day):
        n += 1
    return n