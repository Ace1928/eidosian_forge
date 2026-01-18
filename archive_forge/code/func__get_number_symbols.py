from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _get_number_symbols(locale: Locale | str | None, *, numbering_system: Literal['default'] | str='latn') -> LocaleDataDict:
    parsed_locale = Locale.parse(locale)
    numbering_system = _get_numbering_system(parsed_locale, numbering_system)
    try:
        return parsed_locale.number_symbols[numbering_system]
    except KeyError as error:
        raise UnsupportedNumberingSystemError(f'Unknown numbering system {numbering_system} for Locale {parsed_locale}.') from error