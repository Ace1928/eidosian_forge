from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class TooManySymbolOrPunctuationPlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._punctuation_count = 0
        self._symbol_count = 0
        self._character_count = 0
        self._last_printable_char = None
        self._frenzy_symbol_in_word = False

    def eligible(self, character: str) -> bool:
        return character.isprintable()

    def feed(self, character: str) -> None:
        self._character_count += 1
        if character != self._last_printable_char and character not in COMMON_SAFE_ASCII_CHARACTERS:
            if is_punctuation(character):
                self._punctuation_count += 1
            elif character.isdigit() is False and is_symbol(character) and (is_emoticon(character) is False):
                self._symbol_count += 2
        self._last_printable_char = character

    def reset(self) -> None:
        self._punctuation_count = 0
        self._character_count = 0
        self._symbol_count = 0

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        ratio_of_punctuation = (self._punctuation_count + self._symbol_count) / self._character_count
        return ratio_of_punctuation if ratio_of_punctuation >= 0.3 else 0.0