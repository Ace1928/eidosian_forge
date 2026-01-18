import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
@classmethod
def FromMicroTimestamp(cls, ts):
    """Create new Timestamp object from microsecond UTC timestamp value.

    Args:
      ts: integer microsecond UTC timestamp
    Returns:
      New cls()
    """
    return cls.utcfromtimestamp(ts / _MICROSECONDS_PER_SECOND_F)