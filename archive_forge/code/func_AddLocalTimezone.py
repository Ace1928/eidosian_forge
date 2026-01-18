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
def AddLocalTimezone(cls, obj):
    """If obj is naive, add local timezone to it."""
    if not obj.tzinfo:
        return obj.replace(tzinfo=cls.LocalTimezone)
    return obj