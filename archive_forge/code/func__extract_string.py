from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _extract_string(self, delimiter: str, escapes: t.Optional[t.Set[str]]=None, unescape_sequences: bool=True, raise_unmatched: bool=True) -> str:
    text = ''
    delim_size = len(delimiter)
    escapes = self._STRING_ESCAPES if escapes is None else escapes
    while True:
        if unescape_sequences and self.dialect.UNESCAPED_SEQUENCES and self._peek and (self._char in self.STRING_ESCAPES):
            unescaped_sequence = self.dialect.UNESCAPED_SEQUENCES.get(self._char + self._peek)
            if unescaped_sequence:
                self._advance(2)
                text += unescaped_sequence
                continue
        if self._char in escapes and (self._peek == delimiter or self._peek in escapes) and (self._char not in self._QUOTES or self._char == self._peek):
            if self._peek == delimiter:
                text += self._peek
            else:
                text += self._char + self._peek
            if self._current + 1 < self.size:
                self._advance(2)
            else:
                raise TokenError(f'Missing {delimiter} from {self._line}:{self._current}')
        else:
            if self._chars(delim_size) == delimiter:
                if delim_size > 1:
                    self._advance(delim_size - 1)
                break
            if self._end:
                if not raise_unmatched:
                    return text + self._char
                raise TokenError(f'Missing {delimiter} from {self._line}:{self._start}')
            current = self._current - 1
            self._advance(alnum=True)
            text += self.sql[current:self._current - 1]
    return text