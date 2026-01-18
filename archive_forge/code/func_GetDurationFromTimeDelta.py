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
def GetDurationFromTimeDelta(delta, calendar=False):
    """Returns a Duration object converted from a datetime.timedelta object.

  Args:
    delta: The datetime.timedelta object to convert.
    calendar: Use duration units larger than hours if True.

  Returns:
    The iso_duration.Duration object converted from a datetime.timedelta object.
  """
    return iso_duration.Duration(delta=delta, calendar=calendar)