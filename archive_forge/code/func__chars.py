from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _chars(self, size: int) -> str:
    if size == 1:
        return self._char
    start = self._current - 1
    end = start + size
    return self.sql[start:end] if end <= self.size else ''