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
class label_currency(label_number):
    """
    Labelling currencies

    Parameters
    ----------
    prefix : str
        What to put before the value.

    Examples
    --------
    >>> x = [1.232, 99.2334, 4.6, 9, 4500]
    >>> label_currency()(x)
    ['$1.23', '$99.23', '$4.60', '$9.00', '$4500.00']
    >>> label_currency(prefix='C$', precision=0, big_mark=',')(x)
    ['C$1', 'C$99', 'C$5', 'C$9', 'C$4,500']
    """
    prefix: str = '$'

    def __post_init__(self):
        if self.precision is None and self.accuracy is None:
            self.precision = 2
        super().__post_init__()