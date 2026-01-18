import calendar
import datetime
import logging
import os
import time
import warnings
from tzlocal import windows_tz
def _tz_from_env(tzenv=None):
    if tzenv is None:
        tzenv = os.environ.get('TZ')
    if not tzenv:
        return None
    if tzenv[0] == ':':
        tzenv = tzenv[1:]
    if os.path.isabs(tzenv) and os.path.exists(tzenv):
        tzname = _tz_name_from_env(tzenv)
        if not tzname:
            tzname = tzenv.split(os.sep)[-1]
        with open(tzenv, 'rb') as tzfile:
            return zoneinfo.ZoneInfo.from_file(tzfile, key=tzname)
    try:
        tz = zoneinfo.ZoneInfo(tzenv)
        return tz
    except zoneinfo.ZoneInfoNotFoundError:
        raise zoneinfo.ZoneInfoNotFoundError(f'tzlocal() does not support non-zoneinfo timezones like {tzenv}. \nPlease use a timezone in the form of Continent/City') from None