from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
class _TextAdjustment:

    def __init__(self) -> None:
        self.encoding = get_option('display.encoding')

    def len(self, text: str) -> int:
        return len(text)

    def justify(self, texts: Any, max_len: int, mode: str='right') -> list[str]:
        """
        Perform ljust, center, rjust against string or list-like
        """
        if mode == 'left':
            return [x.ljust(max_len) for x in texts]
        elif mode == 'center':
            return [x.center(max_len) for x in texts]
        else:
            return [x.rjust(max_len) for x in texts]

    def adjoin(self, space: int, *lists, **kwargs) -> str:
        return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)