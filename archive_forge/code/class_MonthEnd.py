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
class MonthEnd(BaseCFTimeOffset):
    _freq = 'ME'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, _days_in_month(other))
        return _shift_month(other, n, 'end')

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date)