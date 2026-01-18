import datetime
import time
import collections.abc
from _sqlite3 import *
def adapt_datetime(val):
    return val.isoformat(' ')