from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
@staticmethod
def _GetPreviousDateTime(t, target_time, tzinfo):
    """Returns the latest datetime <= 't' that has the time target_time.

    Arguments:
      t: a datetime.datetime object, in any timezone
      target_time: a datetime.time object, in any timezone
      tzinfo: a pytz timezone object, or None

    Returns:
      a datetime.datetime object, in the timezone 'tzinfo'
    """
    date = t.date()
    while True:
        result = IntervalTimeSpecification._CombineDateAndTime(date, target_time, tzinfo)
        if result <= t:
            return result
        date -= datetime.timedelta(days=1)