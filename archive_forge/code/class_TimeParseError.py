import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
class TimeParseError(TimestampError):
    """This error is raised when we can't parse the input."""
    pass