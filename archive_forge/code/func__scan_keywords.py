from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _scan_keywords(self) -> None:
    size = 0
    word = None
    chars = self._text
    char = chars
    prev_space = False
    skip = False
    trie = self._KEYWORD_TRIE
    single_token = char in self.SINGLE_TOKENS
    while chars:
        if skip:
            result = TrieResult.PREFIX
        else:
            result, trie = in_trie(trie, char.upper())
        if result == TrieResult.FAILED:
            break
        if result == TrieResult.EXISTS:
            word = chars
        end = self._current + size
        size += 1
        if end < self.size:
            char = self.sql[end]
            single_token = single_token or char in self.SINGLE_TOKENS
            is_space = char.isspace()
            if not is_space or not prev_space:
                if is_space:
                    char = ' '
                chars += char
                prev_space = is_space
                skip = False
            else:
                skip = True
        else:
            char = ''
            break
    if word:
        if self._scan_string(word):
            return
        if self._scan_comment(word):
            return
        if prev_space or single_token or (not char):
            self._advance(size - 1)
            word = word.upper()
            self._add(self.KEYWORDS[word], text=word)
            return
    if self._char in self.SINGLE_TOKENS:
        self._add(self.SINGLE_TOKENS[self._char], text=self._char)
        return
    self._scan_var()