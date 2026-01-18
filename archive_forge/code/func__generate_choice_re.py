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
@staticmethod
def _generate_choice_re(choices: Iterable[str], flags: Union[int, re.RegexFlag]=0) -> Pattern[str]:
    return re.compile('({})'.format('|'.join(choices)), flags=flags)