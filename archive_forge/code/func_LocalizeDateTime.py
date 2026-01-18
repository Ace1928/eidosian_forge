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
def LocalizeDateTime(dt, tzinfo=LOCAL):
    """Returns a datetime object localized to the timezone tzinfo.

  Args:
    dt: The datetime object to localize. It can be timezone naive or aware.
    tzinfo: The timezone of the localized dt. If None then the result is naive,
      otherwise it is aware.

  Returns:
    A datetime object localized to the timezone tzinfo.
  """
    ts = GetTimeStampFromDateTime(dt, tzinfo=tzinfo)
    return GetDateTimeFromTimeStamp(ts, tzinfo=tzinfo)