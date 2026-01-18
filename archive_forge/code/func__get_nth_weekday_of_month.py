from __future__ import absolute_import, print_function, division
import traceback as _traceback
import copy
import math
import re
import sys
import inspect
from time import time
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import calendar
import binascii
import random
import pytz  # noqa
@staticmethod
def _get_nth_weekday_of_month(year, month, day_of_week):
    """ For a given year/month return a list of days in nth-day-of-month order.
        The last weekday of the month is always [-1].
        """
    w = (day_of_week + 6) % 7
    c = calendar.Calendar(w).monthdayscalendar(year, month)
    if c[0][0] == 0:
        c.pop(0)
    return tuple((i[0] for i in c))