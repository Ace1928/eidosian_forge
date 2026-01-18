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
def GetDateTimeDefaults(tzinfo=LOCAL):
    """Returns a datetime object of default values for parsing partial datetimes.

  The year, month and day default to today (right now), and the hour, minute,
  second and fractional second values default to 0.

  Args:
    tzinfo: The timezone of the localized dt. If None then the result is naive,
      otherwise it is aware.

  Returns:
    A datetime object of default values for parsing partial datetimes.
  """
    return datetime.datetime.combine(Now(tzinfo=tzinfo).date(), datetime.time.min)