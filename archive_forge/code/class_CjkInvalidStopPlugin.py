from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class CjkInvalidStopPlugin(MessDetectorPlugin):
    """
    GB(Chinese) based encoding often render the stop incorrectly when the content does not fit and
    can be easily detected. Searching for the overuse of '丅' and '丄'.
    """

    def __init__(self) -> None:
        self._wrong_stop_count = 0
        self._cjk_character_count = 0

    def eligible(self, character: str) -> bool:
        return True

    def feed(self, character: str) -> None:
        if character in {'丅', '丄'}:
            self._wrong_stop_count += 1
            return
        if is_cjk(character):
            self._cjk_character_count += 1

    def reset(self) -> None:
        self._wrong_stop_count = 0
        self._cjk_character_count = 0

    @property
    def ratio(self) -> float:
        if self._cjk_character_count < 16:
            return 0.0
        return self._wrong_stop_count / self._cjk_character_count