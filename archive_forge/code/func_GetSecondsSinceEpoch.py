import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def GetSecondsSinceEpoch(time_tuple):
    """Convert time_tuple (in UTC) to seconds (also in UTC).

  Args:
    time_tuple: tuple with at least 6 items.
  Returns:
    seconds.
  """
    return calendar.timegm(time_tuple[:6] + (0, 0, 0))