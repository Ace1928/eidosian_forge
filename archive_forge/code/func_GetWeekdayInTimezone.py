from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from dateutil import parser
from dateutil import tz
from dateutil.tz import _common as tz_common
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times_data
import six
def GetWeekdayInTimezone(dt, weekday, tzinfo=LOCAL):
    """Returns the Weekday for dt in the timezone specified by tzinfo.

  Args:
    dt: The datetime object that represents the time on weekday.
    weekday: The day of the week specified as a Weekday enum.
    tzinfo: The timezone in which to get the new day of the week in.

  Returns:
    A Weekday that corresponds to dt and weekday pair localized to the timezone
    specified by dt.
  """
    localized_dt = LocalizeDateTime(dt, tzinfo)
    localized_weekday_offset = dt.weekday() - localized_dt.weekday()
    localized_weekday_index = (weekday.value - localized_weekday_offset) % 7
    return Weekday(localized_weekday_index)