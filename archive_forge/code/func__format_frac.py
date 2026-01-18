from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _format_frac(self, value: str, locale: Locale | str | None, force_frac: tuple[int, int] | None=None, *, numbering_system: Literal['default'] | str) -> str:
    min, max = force_frac or self.frac_prec
    if len(value) < min:
        value += '0' * (min - len(value))
    if max == 0 or (min == 0 and int(value) == 0):
        return ''
    while len(value) > min and value[-1] == '0':
        value = value[:-1]
    return get_decimal_symbol(locale, numbering_system=numbering_system) + value