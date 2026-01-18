from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
class datetime_trans(trans):
    """
    Datetime Transformation

    Parameters
    ----------
    tz : str | ZoneInfo
        Timezone information

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> UTC = ZoneInfo("UTC")
    >>> EST = ZoneInfo("EST")
    >>> t = datetime_trans(EST)
    >>> x = [datetime(2022, 1, 20, tzinfo=UTC)]
    >>> x2 = t.inverse(t.transform(x))
    >>> list(x) == list(x2)
    True
    >>> x[0].tzinfo == x2[0].tzinfo
    False
    >>> x[0].tzinfo.key
    'UTC'
    >>> x2[0].tzinfo.key
    'EST'
    """
    domain = (datetime(MINYEAR, 1, 1, tzinfo=UTC), datetime(MAXYEAR, 12, 31, tzinfo=UTC))
    breaks_ = staticmethod(breaks_date())
    format = staticmethod(label_date())
    tz = None

    def __init__(self, tz=None, **kwargs):
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        super().__init__(**kwargs)
        self.tz = tz

    def transform(self, x: DatetimeArrayLike) -> NDArrayFloat:
        """
        Transform from date to a numerical format
        """
        if not len(x):
            return np.array([])
        x0 = next(iter(x))
        try:
            tz = x0.tzinfo
        except AttributeError:
            tz = None
        if tz and self.tz is None:
            self.tz = tz
        return datetime_to_num(x)

    def inverse(self, x: FloatArrayLike) -> NDArrayDatetime:
        """
        Transform to date from numerical format
        """
        return num_to_datetime(x, tz=self.tz)

    @property
    def tzinfo(self):
        """
        Alias of `tz`
        """
        return self.tz