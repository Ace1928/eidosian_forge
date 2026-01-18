import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _unparse_datetime(dt, legacy=False):
    """Format a date or datetime object as a pair of (date, time) strings
    in the format required by the NEWNEWS and NEWGROUPS commands.  If a
    date object is passed, the time is assumed to be midnight (00h00).

    The returned representation depends on the legacy flag:
    * if legacy is False (the default):
      date has the YYYYMMDD format and time the HHMMSS format
    * if legacy is True:
      date has the YYMMDD format and time the HHMMSS format.
    RFC 3977 compliant servers should understand both formats; therefore,
    legacy is only needed when talking to old servers.
    """
    if not isinstance(dt, datetime.datetime):
        time_str = '000000'
    else:
        time_str = '{0.hour:02d}{0.minute:02d}{0.second:02d}'.format(dt)
    y = dt.year
    if legacy:
        y = y % 100
        date_str = '{0:02d}{1.month:02d}{1.day:02d}'.format(y, dt)
    else:
        date_str = '{0:04d}{1.month:02d}{1.day:02d}'.format(y, dt)
    return (date_str, time_str)