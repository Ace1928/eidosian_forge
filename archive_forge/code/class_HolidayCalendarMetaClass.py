from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
class HolidayCalendarMetaClass(type):

    def __new__(cls, clsname: str, bases, attrs):
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class