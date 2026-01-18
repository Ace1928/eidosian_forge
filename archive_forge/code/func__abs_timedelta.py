from __future__ import annotations
import collections.abc
import datetime as dt
import math
import typing
from enum import Enum
from functools import total_ordering
from typing import Any
from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma
def _abs_timedelta(delta: dt.timedelta) -> dt.timedelta:
    """Return an "absolute" value for a timedelta, always representing a time distance.

    Args:
        delta (datetime.timedelta): Input timedelta.

    Returns:
        datetime.timedelta: Absolute timedelta.
    """
    if delta.days < 0:
        now = _now()
        return now - (now + delta)
    return delta