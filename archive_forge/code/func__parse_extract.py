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
def _parse_extract(self) -> exp.Extract:
    this = self._parse_function() or self._parse_var() or self._parse_type()
    if self._match(TokenType.FROM):
        return self.expression(exp.Extract, this=this, expression=self._parse_bitwise())
    if not self._match(TokenType.COMMA):
        self.raise_error('Expected FROM or comma after EXTRACT', self._prev)
    return self.expression(exp.Extract, this=this, expression=self._parse_bitwise())