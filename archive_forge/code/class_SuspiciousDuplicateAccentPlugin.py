from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class SuspiciousDuplicateAccentPlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._successive_count = 0
        self._character_count = 0
        self._last_latin_character = None

    def eligible(self, character: str) -> bool:
        return character.isalpha() and is_latin(character)

    def feed(self, character: str) -> None:
        self._character_count += 1
        if self._last_latin_character is not None and is_accentuated(character) and is_accentuated(self._last_latin_character):
            if character.isupper() and self._last_latin_character.isupper():
                self._successive_count += 1
            if remove_accent(character) == remove_accent(self._last_latin_character):
                self._successive_count += 1
        self._last_latin_character = character

    def reset(self) -> None:
        self._successive_count = 0
        self._character_count = 0
        self._last_latin_character = None

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        return self._successive_count * 2 / self._character_count