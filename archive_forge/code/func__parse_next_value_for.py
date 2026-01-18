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
def _parse_next_value_for(self) -> t.Optional[exp.Expression]:
    if not self._match_text_seq('VALUE', 'FOR'):
        self._retreat(self._index - 1)
        return None
    return self.expression(exp.NextValueFor, this=self._parse_column(), order=self._match(TokenType.OVER) and self._parse_wrapped(self._parse_order))