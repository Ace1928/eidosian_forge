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
def _StringToTime(cls, timestring, tz=None):
    """Use dateutil.parser to convert string into timestamp.

    dateutil.parser understands ISO8601 which is really handy.

    Args:
      timestring: string with datetime
      tz: optional timezone, if timezone is omitted from timestring.
    Returns:
      New Timestamp.
    """
    r = parser.parse(timestring)
    if not r.tzinfo:
        r = (tz or cls.LocalTimezone).localize(r)
    result = cls(r.year, r.month, r.day, r.hour, r.minute, r.second, r.microsecond, r.tzinfo)
    return result