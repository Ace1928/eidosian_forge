from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def TimeDelta_or_None(s):
    try:
        h, m, s = s.split(':')
        if '.' in s:
            s, ms = s.split('.')
            ms = ms.ljust(6, '0')
        else:
            ms = 0
        if h[0] == '-':
            negative = True
        else:
            negative = False
        h, m, s, ms = (abs(int(h)), int(m), int(s), int(ms))
        td = timedelta(hours=h, minutes=m, seconds=s, microseconds=ms)
        if negative:
            return -td
        else:
            return td
    except ValueError:
        return None