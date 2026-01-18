import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
def __calc_month(self):
    a_month = [calendar.month_abbr[i].lower() for i in range(13)]
    f_month = [calendar.month_name[i].lower() for i in range(13)]
    self.a_month = a_month
    self.f_month = f_month