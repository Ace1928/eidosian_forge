from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
class USFederalHolidayCalendar(AbstractHolidayCalendar):
    """
    US Federal Government Holiday Calendar based on rules specified by:
    https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
    """
    rules = [Holiday("New Year's Day", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr, USPresidentsDay, USMemorialDay, Holiday('Juneteenth National Independence Day', month=6, day=19, start_date='2021-06-18', observance=nearest_workday), Holiday('Independence Day', month=7, day=4, observance=nearest_workday), USLaborDay, USColumbusDay, Holiday('Veterans Day', month=11, day=11, observance=nearest_workday), USThanksgivingDay, Holiday('Christmas Day', month=12, day=25, observance=nearest_workday)]