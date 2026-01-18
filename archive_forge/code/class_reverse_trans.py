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
class reverse_trans(trans):
    """
    Reverse Transformation
    """
    transform_is_linear = True
    transform = staticmethod(np.negative)
    inverse = staticmethod(np.negative)