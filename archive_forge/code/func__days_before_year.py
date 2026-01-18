import time as _time
import math as _math
import sys
from operator import index as _index
def _days_before_year(year):
    """year -> number of days before January 1st of year."""
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400