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
class symlog_trans(trans):
    """
    Symmetric Log Transformation

    They symmetric logarithmic transformation is defined as

    ::

        f(x) = log(x+1) for x >= 0
               -log(-x+1) for x < 0

    It can be useful for data that has a wide range of both positive
    and negative values (including zero).
    """
    breaks_: BreaksFunction = breaks_symlog()

    @staticmethod
    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return np.sign(x) * np.log1p(np.abs(x))

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        return np.sign(x) * (np.exp(np.abs(x)) - 1)