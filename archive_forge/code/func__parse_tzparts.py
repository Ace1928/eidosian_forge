import builtins
import datetime
import decimal
from http import client as http_client
import pytz
import re
def _parse_tzparts(parts):
    if 'tz_z' in parts and parts['tz_z'] == 'Z':
        return pytz.UTC
    if 'tz_min' not in parts or not parts['tz_min']:
        return None
    tz_minute_offset = int(parts['tz_hour']) * 60 + int(parts['tz_min'])
    tz_multiplier = -1 if parts['tz_sign'] == '-' else 1
    return pytz.FixedOffset(tz_multiplier * tz_minute_offset)