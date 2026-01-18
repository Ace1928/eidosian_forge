import sys
import datetime
import os.path
from pytz.exceptions import AmbiguousTimeError
from pytz.exceptions import InvalidTimeError
from pytz.exceptions import NonExistentTimeError
from pytz.exceptions import UnknownTimeZoneError
from pytz.lazy import LazyDict, LazyList, LazySet  # noqa
from pytz.tzinfo import unpickler, BaseTzInfo
from pytz.tzfile import build_tzinfo
def _case_insensitive_zone_lookup(zone):
    """case-insensitively matching timezone, else return zone unchanged"""
    global _all_timezones_lower_to_standard
    if _all_timezones_lower_to_standard is None:
        _all_timezones_lower_to_standard = dict(((tz.lower(), tz) for tz in all_timezones))
    return _all_timezones_lower_to_standard.get(zone.lower()) or zone