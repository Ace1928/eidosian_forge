import logging
import re
from typing import (
from . import settings
from .utils import choplist
def _parse_string_1(self, s: bytes, i: int) -> int:
    """Parse literal strings

        PDF Reference 3.2.3
        """
    c = s[i:i + 1]
    if OCT_STRING.match(c) and len(self.oct) < 3:
        self.oct += c
        return i + 1
    elif self.oct:
        self._curtoken += bytes((int(self.oct, 8),))
        self._parse1 = self._parse_string
        return i
    elif c in ESC_STRING:
        self._curtoken += bytes((ESC_STRING[c],))
    elif c == b'\r' and len(s) > i + 1 and (s[i + 1:i + 2] == b'\n'):
        i += 1
    self._parse1 = self._parse_string
    return i + 1