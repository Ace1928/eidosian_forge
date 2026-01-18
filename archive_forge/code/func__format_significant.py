from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _format_significant(self, value: decimal.Decimal, minimum: int, maximum: int) -> str:
    exp = value.adjusted()
    scale = maximum - 1 - exp
    digits = str(value.scaleb(scale).quantize(decimal.Decimal(1)))
    if scale <= 0:
        result = digits + '0' * -scale
    else:
        intpart = digits[:-scale]
        i = len(intpart)
        j = i + max(minimum - i, 0)
        result = '{intpart}.{pad:0<{fill}}{fracpart}{fracextra}'.format(intpart=intpart or '0', pad='', fill=-min(exp + 1, 0), fracpart=digits[i:j], fracextra=digits[j:].rstrip('0')).rstrip('.')
    return result