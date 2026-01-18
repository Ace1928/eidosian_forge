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
def FormatDateTimeList(dt_list, fmt=None, tzinfo=None):
    """Returns a list of strings of datetime objects formatted by FormatDateTime.

  It will use FormatDateTime to process each datetime object in the list.

  Args:
    dt_list: A list of datetime objects to be formatted.
    fmt: The strftime(3) format string, None for the RFC 3339 format in the dt
      timezone ('%Y-%m-%dT%H:%M:%S.%3f%Ez').
    tzinfo: Format dt relative to this timezone.

  Raises:
    DateTimeValueError: A DateTime numeric constant exceeded its range.

  Returns:
    A list of strings of a datetime objects formatted by an extended strftime().
  """
    return [FormatDateTime(dt, fmt, tzinfo) for dt in dt_list]