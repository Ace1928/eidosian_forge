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
class YearOffset(BaseCFTimeOffset):
    _freq: ClassVar[str]
    _day_option: ClassVar[str]
    _default_month: ClassVar[int]

    def __init__(self, n=1, month=None):
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        reference_day = _get_day_of_month(other, self._day_option)
        years = _adjust_n_years(other, self.n, self.month, reference_day)
        months = years * 12 + (self.month - other.month)
        return _shift_month(other, months, self._day_option)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        return f'{self._freq}-{_MONTH_ABBREVIATIONS[self.month]}'

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'