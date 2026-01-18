from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class SuperWeirdWordPlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._word_count = 0
        self._bad_word_count = 0
        self._foreign_long_count = 0
        self._is_current_word_bad = False
        self._foreign_long_watch = False
        self._character_count = 0
        self._bad_character_count = 0
        self._buffer = ''
        self._buffer_accent_count = 0

    def eligible(self, character: str) -> bool:
        return True

    def feed(self, character: str) -> None:
        if character.isalpha():
            self._buffer = ''.join([self._buffer, character])
            if is_accentuated(character):
                self._buffer_accent_count += 1
            if self._foreign_long_watch is False and (is_latin(character) is False or is_accentuated(character)) and (is_cjk(character) is False) and (is_hangul(character) is False) and (is_katakana(character) is False) and (is_hiragana(character) is False) and (is_thai(character) is False):
                self._foreign_long_watch = True
            return
        if not self._buffer:
            return
        if (character.isspace() or is_punctuation(character) or is_separator(character)) and self._buffer:
            self._word_count += 1
            buffer_length = len(self._buffer)
            self._character_count += buffer_length
            if buffer_length >= 4:
                if self._buffer_accent_count / buffer_length > 0.34:
                    self._is_current_word_bad = True
                if is_accentuated(self._buffer[-1]) and self._buffer[-1].isupper():
                    self._foreign_long_count += 1
                    self._is_current_word_bad = True
            if buffer_length >= 24 and self._foreign_long_watch:
                self._foreign_long_count += 1
                self._is_current_word_bad = True
            if self._is_current_word_bad:
                self._bad_word_count += 1
                self._bad_character_count += len(self._buffer)
                self._is_current_word_bad = False
            self._foreign_long_watch = False
            self._buffer = ''
            self._buffer_accent_count = 0
        elif character not in {'<', '>', '-', '=', '~', '|', '_'} and character.isdigit() is False and is_symbol(character):
            self._is_current_word_bad = True
            self._buffer += character

    def reset(self) -> None:
        self._buffer = ''
        self._is_current_word_bad = False
        self._foreign_long_watch = False
        self._bad_word_count = 0
        self._word_count = 0
        self._character_count = 0
        self._bad_character_count = 0
        self._foreign_long_count = 0

    @property
    def ratio(self) -> float:
        if self._word_count <= 10 and self._foreign_long_count == 0:
            return 0.0
        return self._bad_character_count / self._character_count