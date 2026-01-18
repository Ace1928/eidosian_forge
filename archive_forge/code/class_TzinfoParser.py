import re
import sys
from datetime import datetime, timedelta
from datetime import tzinfo as dt_tzinfo
from functools import lru_cache
from typing import (
from dateutil import tz
from arrow import locales
from arrow.constants import DEFAULT_LOCALE
from arrow.util import next_weekday, normalize_timestamp
class TzinfoParser:
    _TZINFO_RE: ClassVar[Pattern[str]] = re.compile('^(?:\\(UTC)*([\\+\\-])?(\\d{2})(?:\\:?(\\d{2}))?')

    @classmethod
    def parse(cls, tzinfo_string: str) -> dt_tzinfo:
        tzinfo: Optional[dt_tzinfo] = None
        if tzinfo_string == 'local':
            tzinfo = tz.tzlocal()
        elif tzinfo_string in ['utc', 'UTC', 'Z']:
            tzinfo = tz.tzutc()
        else:
            iso_match = cls._TZINFO_RE.match(tzinfo_string)
            if iso_match:
                sign: Optional[str]
                hours: str
                minutes: Union[str, int, None]
                sign, hours, minutes = iso_match.groups()
                seconds = int(hours) * 3600 + int(minutes or 0) * 60
                if sign == '-':
                    seconds *= -1
                tzinfo = tz.tzoffset(None, seconds)
            else:
                tzinfo = tz.gettz(tzinfo_string)
        if tzinfo is None:
            raise ParserError(f'Could not parse timezone expression {tzinfo_string!r}.')
        return tzinfo