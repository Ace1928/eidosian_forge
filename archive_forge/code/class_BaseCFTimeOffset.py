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
class BaseCFTimeOffset:
    _freq: ClassVar[str | None] = None
    _day_option: ClassVar[str | None] = None

    def __init__(self, n: int=1):
        if not isinstance(n, int):
            raise TypeError(f"The provided multiple 'n' must be an integer. Instead a value of type {type(n)!r} was provided.")
        self.n = n

    def rule_code(self):
        return self._freq

    def __eq__(self, other):
        return self.n == other.n and self.rule_code() == other.rule_code()

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        return self.__apply__(other)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract a cftime.datetime from a time offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n)
        else:
            return NotImplemented

    def __mul__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(n=other * self.n)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, BaseCFTimeOffset) and type(self) != type(other):
            raise TypeError('Cannot subtract cftime offsets of differing types')
        return -self + other

    def __apply__(self):
        return NotImplemented

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        test_date = self + date - self
        return date == test_date

    def rollforward(self, date):
        if self.onOffset(date):
            return date
        else:
            return date + type(self)()

    def rollback(self, date):
        if self.onOffset(date):
            return date
        else:
            return date - type(self)()

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}>'

    def __repr__(self):
        return str(self)

    def _get_offset_day(self, other):
        return _get_day_of_month(other, self._day_option)