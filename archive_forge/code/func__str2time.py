import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _str2time(day, mon, yr, hr, min, sec, tz):
    yr = int(yr)
    if yr > datetime.MAXYEAR:
        return None
    try:
        mon = MONTHS_LOWER.index(mon.lower()) + 1
    except ValueError:
        try:
            imon = int(mon)
        except ValueError:
            return None
        if 1 <= imon <= 12:
            mon = imon
        else:
            return None
    if hr is None:
        hr = 0
    if min is None:
        min = 0
    if sec is None:
        sec = 0
    day = int(day)
    hr = int(hr)
    min = int(min)
    sec = int(sec)
    if yr < 1000:
        cur_yr = time.localtime(time.time())[0]
        m = cur_yr % 100
        tmp = yr
        yr = yr + cur_yr - m
        m = m - tmp
        if abs(m) > 50:
            if m > 0:
                yr = yr + 100
            else:
                yr = yr - 100
    t = _timegm((yr, mon, day, hr, min, sec, tz))
    if t is not None:
        if tz is None:
            tz = 'UTC'
        tz = tz.upper()
        offset = offset_from_tz_string(tz)
        if offset is None:
            return None
        t = t - offset
    return t