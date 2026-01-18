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
def _parse_factor(self) -> t.Optional[exp.Expression]:
    parse_method = self._parse_exponent if self.EXPONENT else self._parse_unary
    this = parse_method()
    while self._match_set(self.FACTOR):
        this = self.expression(self.FACTOR[self._prev.token_type], this=this, comments=self._prev_comments, expression=parse_method())
        if isinstance(this, exp.Div):
            this.args['typed'] = self.dialect.TYPED_DIVISION
            this.args['safe'] = self.dialect.SAFE_DIVISION
    return this