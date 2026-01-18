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
def _set_locale(self, locale: Locale | str | None) -> None:
    if locale is None:
        self._locale_identifier = None
        self._locale = None
        return
    if isinstance(locale, Locale):
        self._locale_identifier = str(locale)
        self._locale = locale
        return
    if isinstance(locale, str):
        self._locale_identifier = str(locale)
        try:
            self._locale = Locale.parse(locale)
        except UnknownLocaleError:
            self._locale = None
        return
    raise TypeError(f'`locale` must be a Locale, a locale identifier string, or None; got {locale!r}')