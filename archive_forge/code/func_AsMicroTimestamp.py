import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def AsMicroTimestamp(self):
    """Return microsecond timestamp constructed from this object."""
    return SecondsToMicroseconds(self.AsSecondsSinceEpoch()) + self.microsecond