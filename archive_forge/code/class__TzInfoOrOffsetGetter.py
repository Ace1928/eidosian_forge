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
class _TzInfoOrOffsetGetter(object):
    """A helper class for dateutil.parser.parse().

  Attributes:
    _timezone_was_specified: True if the parsed date/time string contained
      an explicit timezone name or offset.
  """

    def __init__(self):
        self._timezone_was_specified = False

    def Get(self, name, offset):
        """Returns the tzinfo for name or offset.

    Used by dateutil.parser.parse() to convert timezone names and offsets.

    Args:
      name: A timezone name or None to use offset. If offset is also None then
        the local tzinfo is returned.
      offset: A signed UTC timezone offset in seconds.

    Returns:
      The tzinfo for name or offset or the local tzinfo if both are None.
    """
        if name or offset:
            self._timezone_was_specified = True
        if not name and offset is not None:
            return offset
        return GetTimeZone(name)

    @property
    def timezone_was_specified(self):
        """True if the parsed date/time string contained an explicit timezone."""
        return self._timezone_was_specified