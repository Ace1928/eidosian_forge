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
class pd_timedelta_trans(trans):
    """
    Pandas timedelta Transformation
    """
    domain = (pd.Timedelta.min, pd.Timedelta.max)
    breaks_ = staticmethod(breaks_timedelta())
    format = staticmethod(label_timedelta())

    @staticmethod
    def transform(x: TimedeltaSeries) -> NDArrayFloat:
        """
        Transform from Timeddelta to numerical format
        """
        return np.array([_x.value for _x in x])

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayTimedelta:
        """
        Transform to Timedelta from numerical format
        """
        return np.array([pd.Timedelta(int(i)) for i in x])