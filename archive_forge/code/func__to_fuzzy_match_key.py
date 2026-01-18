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
def _to_fuzzy_match_key(self, key: tuple[str, str] | str) -> str:
    """Converts a message key to a string suitable for fuzzy matching."""
    if isinstance(key, tuple):
        matchkey = key[0]
    else:
        matchkey = key
    return matchkey.lower().strip()