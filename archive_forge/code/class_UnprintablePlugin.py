from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class UnprintablePlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._unprintable_count = 0
        self._character_count = 0

    def eligible(self, character: str) -> bool:
        return True

    def feed(self, character: str) -> None:
        if character.isspace() is False and character.isprintable() is False and (character != '\x1a'):
            self._unprintable_count += 1
        self._character_count += 1

    def reset(self) -> None:
        self._unprintable_count = 0

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        return self._unprintable_count * 8 / self._character_count