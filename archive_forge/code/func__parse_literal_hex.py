import logging
import re
from typing import (
from . import settings
from .utils import choplist
def _parse_literal_hex(self, s: bytes, i: int) -> int:
    c = s[i:i + 1]
    if HEX.match(c) and len(self.hex) < 2:
        self.hex += c
        return i + 1
    if self.hex:
        self._curtoken += bytes((int(self.hex, 16),))
    self._parse1 = self._parse_literal
    return i