from datetime import timedelta, time, date
from time import localtime
def _month(val):
    for key, mon in _str2num.items():
        if key in val:
            return mon
    raise TypeError("unknown month '%s'" % val)