from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _iter_patterns(a_unit):
    if add_direction:
        unit_rel_patterns = locale._data['date_fields'][a_unit]
        if seconds >= 0:
            yield unit_rel_patterns['future']
        else:
            yield unit_rel_patterns['past']
    a_unit = f'duration-{a_unit}'
    yield locale._data['unit_patterns'].get(a_unit, {}).get(format)