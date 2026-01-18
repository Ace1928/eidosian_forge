from calendar import timegm
from datetime import datetime, time, timedelta
from email.utils import parsedate_tz, mktime_tz
import re
import aniso8601
import pytz
class int_range(object):
    """ Restrict input to an integer in a range (inclusive) """

    def __init__(self, low, high, argument='argument'):
        self.low = low
        self.high = high
        self.argument = argument

    def __call__(self, value):
        value = _get_integer(value)
        if value < self.low or value > self.high:
            error = 'Invalid {arg}: {val}. {arg} must be within the range {lo} - {hi}'.format(arg=self.argument, val=value, lo=self.low, hi=self.high)
            raise ValueError(error)
        return value