from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _quantize_value(self, value: decimal.Decimal, locale: Locale | str | None, frac_prec: tuple[int, int], group_separator: bool, *, numbering_system: Literal['default'] | str) -> str:
    if value.is_infinite():
        return get_infinity_symbol(locale, numbering_system=numbering_system)
    quantum = get_decimal_quantum(frac_prec[1])
    rounded = value.quantize(quantum)
    a, sep, b = f'{rounded:f}'.partition('.')
    integer_part = a
    if group_separator:
        integer_part = self._format_int(a, self.int_prec[0], self.int_prec[1], locale, numbering_system=numbering_system)
    number = integer_part + self._format_frac(b or '0', locale=locale, force_frac=frac_prec, numbering_system=numbering_system)
    return number