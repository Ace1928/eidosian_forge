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
class label_ordinal:
    """
    Ordinal number labelling

    Parameters
    ----------
    prefix : str
        What to put before the value.
    suffix : str
        What to put after the value.
    big_mark : str
        The thousands separator. This is usually
        a comma or a dot.

    Examples
    --------
    >>> label_ordinal()(range(8))
    ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th']
    >>> label_ordinal(suffix=' Number')(range(11, 15))
    ['11th Number', '12th Number', '13th Number', '14th Number']
    """
    prefix: str = ''
    suffix: str = ''
    big_mark: str = ''

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        labels = [ordinal(num, self.prefix, self.suffix, self.big_mark) for num in x]
        return labels