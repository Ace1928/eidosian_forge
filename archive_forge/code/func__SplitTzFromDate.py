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
def _SplitTzFromDate(string):
    """Returns (prefix,tzinfo) if string has a trailing tz, else (None,None)."""
    try:
        match = re.match('(.*[\\d\\s])([^\\d\\s]+)$', string)
    except TypeError:
        return (None, None)
    if match:
        tzinfo = GetTimeZone(match.group(2))
        if tzinfo:
            return (match.group(1), tzinfo)
    return (None, None)