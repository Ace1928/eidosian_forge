from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _convert_quotes(arr: t.List[str | t.Tuple[str, str]]) -> t.Dict[str, str]:
    return dict(((item, item) if isinstance(item, str) else (item[0], item[1]) for item in arr))