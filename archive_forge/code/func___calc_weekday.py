import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
def __calc_weekday(self):
    a_weekday = [calendar.day_abbr[i].lower() for i in range(7)]
    f_weekday = [calendar.day_name[i].lower() for i in range(7)]
    self.a_weekday = a_weekday
    self.f_weekday = f_weekday