from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class SuspiciousRange(MessDetectorPlugin):

    def __init__(self) -> None:
        self._suspicious_successive_range_count = 0
        self._character_count = 0
        self._last_printable_seen = None

    def eligible(self, character: str) -> bool:
        return character.isprintable()

    def feed(self, character: str) -> None:
        self._character_count += 1
        if character.isspace() or is_punctuation(character) or character in COMMON_SAFE_ASCII_CHARACTERS:
            self._last_printable_seen = None
            return
        if self._last_printable_seen is None:
            self._last_printable_seen = character
            return
        unicode_range_a = unicode_range(self._last_printable_seen)
        unicode_range_b = unicode_range(character)
        if is_suspiciously_successive_range(unicode_range_a, unicode_range_b):
            self._suspicious_successive_range_count += 1
        self._last_printable_seen = character

    def reset(self) -> None:
        self._character_count = 0
        self._suspicious_successive_range_count = 0
        self._last_printable_seen = None

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        ratio_of_suspicious_range_usage = self._suspicious_successive_range_count * 2 / self._character_count
        if ratio_of_suspicious_range_usage < 0.1:
            return 0.0
        return ratio_of_suspicious_range_usage