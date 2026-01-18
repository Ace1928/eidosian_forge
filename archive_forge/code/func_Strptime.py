from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import calendar
import datetime
import re
import time
from six.moves import map  # pylint: disable=redefined-builtin
def Strptime(rfc3339_str):
    """Converts an RFC 3339 timestamp to Unix time in seconds since the epoch.

  Args:
    rfc3339_str: a timestamp in RFC 3339 format (yyyy-mm-ddThh:mm:ss.sss
        followed by a time zone, given as Z, +hh:mm, or -hh:mm)

  Returns:
    a number of seconds since January 1, 1970, 00:00:00 UTC

  Raises:
    ValueError: if the timestamp is not in an acceptable format
  """
    match = re.match('(\\d\\d\\d\\d)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)(?:\\.(\\d+))?(?:(Z)|([-+])(\\d\\d):(\\d\\d))', rfc3339_str)
    if not match:
        raise ValueError('not a valid timestamp: %r' % rfc3339_str)
    year, month, day, hour, minute, second, frac_seconds, zulu, zone_sign, zone_hours, zone_minutes = match.groups()
    time_tuple = list(map(int, [year, month, day, hour, minute, second]))
    if zulu == 'Z':
        zone_offset = 0
    else:
        zone_offset = int(zone_hours) * 3600 + int(zone_minutes) * 60
        if zone_sign == '-':
            zone_offset = -zone_offset
    integer_time = calendar.timegm(time_tuple) - zone_offset
    if frac_seconds:
        sig_dig = len(frac_seconds)
        return (integer_time * 10 ** sig_dig + int(frac_seconds)) * 10 ** (-sig_dig)
    else:
        return integer_time