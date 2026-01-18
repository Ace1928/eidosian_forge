import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def SecondsToMicroseconds(seconds):
    """Convert seconds to microseconds.

  Args:
    seconds: number
  Returns:
    microseconds
  """
    return seconds * _MICROSECONDS_PER_SECOND