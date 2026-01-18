from __future__ import annotations
import datetime
import re
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from copy import copy
from difflib import SequenceMatcher
from email import message_from_string
from heapq import nlargest
from typing import TYPE_CHECKING
from babel import __version__ as VERSION
from babel.core import Locale, UnknownLocaleError
from babel.dates import format_datetime
from babel.messages.plurals import get_plural
from babel.util import LOCALTZ, FixedOffsetTimezone, _cmp, distinct
def _parse_datetime_header(value: str) -> datetime.datetime:
    match = re.match('^(?P<datetime>.*?)(?P<tzoffset>[+-]\\d{4})?$', value)
    dt = datetime.datetime.strptime(match.group('datetime'), '%Y-%m-%d %H:%M')
    tzoffset = match.group('tzoffset')
    if tzoffset is not None:
        plus_minus_s, rest = (tzoffset[0], tzoffset[1:])
        hours_offset_s, mins_offset_s = (rest[:2], rest[2:])
        plus_minus = int(f'{plus_minus_s}1')
        hours_offset = int(hours_offset_s)
        mins_offset = int(mins_offset_s)
        net_mins_offset = hours_offset * 60
        net_mins_offset += mins_offset
        net_mins_offset *= plus_minus
        tzoffset = FixedOffsetTimezone(net_mins_offset)
        dt = dt.replace(tzinfo=tzoffset)
    return dt