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
@lru_cache(maxsize=1024)
def _cached_parse_pattern(pattern: str) -> DateTimePattern:
    result = []
    for tok_type, tok_value in tokenize_pattern(pattern):
        if tok_type == 'chars':
            result.append(tok_value.replace('%', '%%'))
        elif tok_type == 'field':
            fieldchar, fieldnum = tok_value
            limit = PATTERN_CHARS[fieldchar]
            if limit and fieldnum not in limit:
                raise ValueError(f'Invalid length for field: {fieldchar * fieldnum!r}')
            result.append('%%(%s)s' % (fieldchar * fieldnum))
        else:
            raise NotImplementedError(f'Unknown token type: {tok_type}')
    return DateTimePattern(pattern, ''.join(result))