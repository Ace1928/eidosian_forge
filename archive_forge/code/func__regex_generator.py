from __future__ import annotations
import re
import typing
from abc import ABC, abstractmethod
from itertools import zip_longest
from operator import attrgetter
from marshmallow import types
from marshmallow.exceptions import ValidationError
def _regex_generator(self, relative: bool, absolute: bool, require_tld: bool) -> typing.Pattern:
    hostname_variants = ['(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\\.)+(?:[A-Z]{2,6}\\.?|[A-Z0-9-]{2,}\\.?)', 'localhost', '\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}', '\\[[A-F0-9]*:[A-F0-9:]+\\]']
    if not require_tld:
        hostname_variants.append('(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\\.?)')
    absolute_part = ''.join(('(?:[a-z0-9\\.\\-\\+]*)://', "(?:(?:[a-z0-9\\-._~!$&'()*+,;=:]|%[0-9a-f]{2})*@)?", '(?:', '|'.join(hostname_variants), ')', '(?::\\d+)?'))
    relative_part = '(?:/?|[/?]\\S+)\\Z'
    if relative:
        if absolute:
            parts: tuple[str, ...] = ('^(', absolute_part, ')?', relative_part)
        else:
            parts = ('^', relative_part)
    else:
        parts = ('^', absolute_part, relative_part)
    return re.compile(''.join(parts), re.IGNORECASE)