import calendar
import datetime
import logging
import os
import time
import warnings
from tzlocal import windows_tz
def _tz_name_from_env(tzenv=None):
    if tzenv is None:
        tzenv = os.environ.get('TZ')
    if not tzenv:
        return None
    log.debug(f'Found a TZ environment: {tzenv}')
    if tzenv[0] == ':':
        tzenv = tzenv[1:]
    if tzenv in windows_tz.tz_win:
        return tzenv
    if os.path.isabs(tzenv) and os.path.exists(tzenv):
        parts = os.path.realpath(tzenv).split(os.sep)
        possible_tz = '/'.join(parts[-2:])
        if possible_tz in windows_tz.tz_win:
            return possible_tz
        if parts[-1] in windows_tz.tz_win:
            return parts[-1]
    log.debug('TZ does not contain a time zone name')
    return None