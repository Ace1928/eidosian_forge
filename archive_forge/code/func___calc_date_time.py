import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
def __calc_date_time(self):
    time_tuple = time.struct_time((1999, 3, 17, 22, 44, 55, 2, 76, 0))
    date_time = [None, None, None]
    date_time[0] = time.strftime('%c', time_tuple).lower()
    date_time[1] = time.strftime('%x', time_tuple).lower()
    date_time[2] = time.strftime('%X', time_tuple).lower()
    replacement_pairs = [('%', '%%'), (self.f_weekday[2], '%A'), (self.f_month[3], '%B'), (self.a_weekday[2], '%a'), (self.a_month[3], '%b'), (self.am_pm[1], '%p'), ('1999', '%Y'), ('99', '%y'), ('22', '%H'), ('44', '%M'), ('55', '%S'), ('76', '%j'), ('17', '%d'), ('03', '%m'), ('3', '%m'), ('2', '%w'), ('10', '%I')]
    replacement_pairs.extend([(tz, '%Z') for tz_values in self.timezone for tz in tz_values])
    for offset, directive in ((0, '%c'), (1, '%x'), (2, '%X')):
        current_format = date_time[offset]
        for old, new in replacement_pairs:
            if old:
                current_format = current_format.replace(old, new)
        time_tuple = time.struct_time((1999, 1, 3, 1, 1, 1, 6, 3, 0))
        if '00' in time.strftime(directive, time_tuple):
            U_W = '%W'
        else:
            U_W = '%U'
        date_time[offset] = current_format.replace('11', U_W)
    self.LC_date_time = date_time[0]
    self.LC_date = date_time[1]
    self.LC_time = date_time[2]