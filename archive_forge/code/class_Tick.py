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
class Tick(BaseCFTimeOffset):

    def _next_higher_resolution(self):
        self_type = type(self)
        if self_type not in [Day, Hour, Minute, Second, Millisecond]:
            raise ValueError('Could not convert to integer offset at any resolution')
        if type(self) is Day:
            return Hour(self.n * 24)
        if type(self) is Hour:
            return Minute(self.n * 60)
        if type(self) is Minute:
            return Second(self.n * 60)
        if type(self) is Second:
            return Millisecond(self.n * 1000)
        if type(self) is Millisecond:
            return Microsecond(self.n * 1000)

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        if isinstance(other, float):
            n = other * self.n
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return type(self)(n=other * self.n)

    def as_timedelta(self):
        """All Tick subclasses must implement an as_timedelta method."""
        raise NotImplementedError