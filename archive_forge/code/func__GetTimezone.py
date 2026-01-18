from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def _GetTimezone(timezone_string):
    """Converts a timezone string to a pytz timezone object.

  Arguments:
    timezone_string: a string representing a timezone, or None

  Returns:
    a pytz timezone object, or None

  Raises:
    ValueError: if timezone_string is specified, but pytz module could not be
        loaded
  """
    if pytz is None:
        if timezone_string:
            raise ValueError('need pytz in order to specify a timezone')
        return None
    if timezone_string:
        return pytz.timezone(timezone_string)
    else:
        return pytz.utc