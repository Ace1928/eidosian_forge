from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
def HolidayCalendarFactory(name: str, base, other, base_class=AbstractHolidayCalendar):
    rules = AbstractHolidayCalendar.merge_class(base, other)
    calendar_class = type(name, (base_class,), {'rules': rules, 'name': name})
    return calendar_class