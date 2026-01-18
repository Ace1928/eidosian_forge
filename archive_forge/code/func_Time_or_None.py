from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def Time_or_None(s):
    try:
        h, m, s = s.split(':')
        if '.' in s:
            s, ms = s.split('.')
            ms = ms.ljust(6, '0')
        else:
            ms = 0
        h, m, s, ms = (int(h), int(m), int(s), int(ms))
        return time(hour=h, minute=m, second=s, microsecond=ms)
    except ValueError:
        return None