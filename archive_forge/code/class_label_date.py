from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_date:
    """
    Datetime labels

    Parameters
    ----------
    fmt : str
        Format string. See
        :ref:`strftime <strftime-strptime-behavior>`.
    tz : datetime.tzinfo, optional
        Time zone information. If none is specified, the
        time zone will be that of the first date. If the
        first date has no time information then a time zone
        is chosen by other means.

    Examples
    --------
    >>> from datetime import datetime
    >>> x = [datetime(x, 1, 1) for x in [2010, 2014, 2018, 2022]]
    >>> label_date()(x)
    ['2010-01-01', '2014-01-01', '2018-01-01', '2022-01-01']
    >>> label_date('%Y')(x)
    ['2010', '2014', '2018', '2022']

    Can format time

    >>> x = [datetime(2017, 12, 1, 16, 5, 7)]
    >>> label_date("%Y-%m-%d %H:%M:%S")(x)
    ['2017-12-01 16:05:07']

    Time zones are respected

    >>> UTC = ZoneInfo('UTC')
    >>> UG = ZoneInfo('Africa/Kampala')
    >>> x = [datetime(2010, 1, 1, i) for i in [8, 15]]
    >>> x_tz = [datetime(2010, 1, 1, i, tzinfo=UG) for i in [8, 15]]
    >>> label_date('%Y-%m-%d %H:%M')(x)
    ['2010-01-01 08:00', '2010-01-01 15:00']
    >>> label_date('%Y-%m-%d %H:%M')(x_tz)
    ['2010-01-01 08:00', '2010-01-01 15:00']

    Format with a specific time zone

    >>> label_date('%Y-%m-%d %H:%M', tz=UTC)(x_tz)
    ['2010-01-01 05:00', '2010-01-01 12:00']
    >>> label_date('%Y-%m-%d %H:%M', tz='EST')(x_tz)
    ['2010-01-01 00:00', '2010-01-01 07:00']
    """
    fmt: str = '%Y-%m-%d'
    tz: Optional[tzinfo] = None

    def __post_init__(self):
        if isinstance(self.tz, str):
            self.tz = ZoneInfo(self.tz)

    def __call__(self, x: Sequence[datetime]) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        if self.tz is not None:
            x = [d.astimezone(self.tz) for d in x]
        return [d.strftime(self.fmt) for d in x]