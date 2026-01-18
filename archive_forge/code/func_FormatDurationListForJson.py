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
def FormatDurationListForJson(duration_list):
    """Returns a list of string representations of the durations, ending in 's'.

  It will use FormatDurationForJson to process each duration object in the list.

  Args:
    duration_list: A list of iso_duration.Duration objects to be formatted.

  Raises:
    DurationValueError: A Duration numeric constant exceeded its range.

  Returns:
    A list of strings representation of the duration.
  """
    return [FormatDurationForJson(duration) for duration in duration_list]