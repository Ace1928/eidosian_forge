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
def _IntStringToInterval(cls, timestring):
    """Parse interval date specification and create a timedelta object."""
    return datetime.timedelta(seconds=ConvertIntervalToSeconds(timestring))