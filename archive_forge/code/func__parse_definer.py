from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def _parse_definer(self) -> t.Optional[exp.DefinerProperty]:
    self._match(TokenType.EQ)
    user = self._parse_id_var()
    self._match(TokenType.PARAMETER)
    host = self._parse_id_var() or (self._match(TokenType.MOD) and self._prev.text)
    if not user or not host:
        return None
    return exp.DefinerProperty(this=f'{user}@{host}')