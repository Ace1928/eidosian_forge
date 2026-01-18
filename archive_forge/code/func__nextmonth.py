import sys
import datetime
import locale as _locale
from itertools import repeat
def _nextmonth(year, month):
    if month == 12:
        return (year + 1, 1)
    else:
        return (year, month + 1)