from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import calendar
import datetime
import re
import time
from six.moves import map  # pylint: disable=redefined-builtin
def CalculateExpiration(num_seconds):
    """Takes a number of seconds and returns the expiration time in RFC 3339."""
    if num_seconds is None:
        return None
    utc_now = CurrentDatetimeUtc()
    adjusted = utc_now + datetime.timedelta(0, int(num_seconds))
    formatted_expiration = _FormatDateString(adjusted)
    return formatted_expiration