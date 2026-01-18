import logging
import re
from typing import (
from . import settings
from .utils import choplist
def _parse_wopen(self, s: bytes, i: int) -> int:
    c = s[i:i + 1]
    if c == b'<':
        self._add_token(KEYWORD_DICT_BEGIN)
        self._parse1 = self._parse_main
        i += 1
    else:
        self._parse1 = self._parse_hexstring
    return i