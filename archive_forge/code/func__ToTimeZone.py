from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def _ToTimeZone(t, tzinfo):
    """Converts 't' to the time zone 'tzinfo'.

  Arguments:
    t: a datetime object.  It may be in any pytz time zone, or it may be
      timezone-naive (interpreted as UTC).
    tzinfo: a pytz timezone object, or None.

  Returns:
    a datetime object in the time zone 'tzinfo'
  """
    if pytz is None:
        return t.replace(tzinfo=None)
    elif tzinfo:
        if not t.tzinfo:
            t = pytz.utc.localize(t)
        return tzinfo.normalize(t.astimezone(tzinfo))
    elif t.tzinfo:
        return pytz.utc.normalize(t.astimezone(pytz.utc)).replace(tzinfo=None)
    else:
        return t