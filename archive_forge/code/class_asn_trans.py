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
class asn_trans(trans):
    """
    Arc-sin square-root Transformation
    """

    @staticmethod
    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return 2 * np.arcsin(np.sqrt(x))

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        return np.sin(x / 2) ** 2