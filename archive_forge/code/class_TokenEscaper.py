from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
class TokenEscaper:
    """
    Escape punctuation within an input string.
    """
    DEFAULT_ESCAPED_CHARS = '[,.<>{}\\[\\]\\\\\\"\\\':;!@#$%^&*()\\-+=~\\/ ]'

    def __init__(self, escape_chars_re: Optional[Pattern]=None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f'Value must be a string object for token escaping.Got type {type(value)}')

        def escape_symbol(match: re.Match) -> str:
            value = match.group(0)
            return f'\\{value}'
        return self.escaped_chars_re.sub(escape_symbol, value)