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
def _parse_sort(self, exp_class: t.Type[E], token: TokenType) -> t.Optional[E]:
    if not self._match(token):
        return None
    return self.expression(exp_class, expressions=self._parse_csv(self._parse_ordered))