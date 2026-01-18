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
def _advance_any(self, ignore_reserved: bool=False) -> t.Optional[Token]:
    if self._curr and (ignore_reserved or self._curr.token_type not in self.RESERVED_TOKENS):
        self._advance()
        return self._prev
    return None