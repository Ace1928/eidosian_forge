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
class label_percent(label_number):
    """
    Labelling percentages

    Multiply by one hundred and display percent sign

    Examples
    --------
    >>> label = label_percent()
    >>> label([.45, 9.515, .01])
    ['45%', '952%', '1%']
    >>> label([.654, .8963, .1])
    ['65%', '90%', '10%']
    """
    scale: float = 100
    suffix: str = '%'