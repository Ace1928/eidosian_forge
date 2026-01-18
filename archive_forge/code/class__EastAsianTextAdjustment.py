from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
class _EastAsianTextAdjustment(_TextAdjustment):

    def __init__(self) -> None:
        super().__init__()
        if get_option('display.unicode.ambiguous_as_wide'):
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1
        self._EAW_MAP = {'Na': 1, 'N': 1, 'W': 2, 'F': 2, 'H': 1}

    def len(self, text: str) -> int:
        """
        Calculate display width considering unicode East Asian Width
        """
        if not isinstance(text, str):
            return len(text)
        return sum((self._EAW_MAP.get(east_asian_width(c), self.ambiguous_width) for c in text))

    def justify(self, texts: Iterable[str], max_len: int, mode: str='right') -> list[str]:

        def _get_pad(t):
            return max_len - self.len(t) + len(t)
        if mode == 'left':
            return [x.ljust(_get_pad(x)) for x in texts]
        elif mode == 'center':
            return [x.center(_get_pad(x)) for x in texts]
        else:
            return [x.rjust(_get_pad(x)) for x in texts]