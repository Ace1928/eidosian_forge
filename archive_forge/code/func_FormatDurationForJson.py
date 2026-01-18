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
def FormatDurationForJson(duration):
    """Returns a string representation of the duration, ending in 's'.

  See the section of
  <https://github.com/google/protobuf/blob/master/src/google/protobuf/duration.proto>
  on JSON formats.

  For example:

    >>> FormatDurationForJson(iso_duration.Duration(seconds=10))
    10s
    >>> FormatDurationForJson(iso_duration.Duration(hours=1))
    3600s
    >>> FormatDurationForJson(iso_duration.Duration(seconds=1, microseconds=5))
    1.000005s

  Args:
    duration: An iso_duration.Duration object.

  Raises:
    DurationValueError: A Duration numeric constant exceeded its range.

  Returns:
    An string representation of the duration.
  """
    num = '{}'.format(round(duration.total_seconds, _MICROSECOND_PRECISION))
    if num.endswith('.0'):
        num = num[:-len('.0')]
    return num + 's'