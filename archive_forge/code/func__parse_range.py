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
def _parse_range(self, this: t.Optional[exp.Expression]=None) -> t.Optional[exp.Expression]:
    this = this or self._parse_bitwise()
    negate = self._match(TokenType.NOT)
    if self._match_set(self.RANGE_PARSERS):
        expression = self.RANGE_PARSERS[self._prev.token_type](self, this)
        if not expression:
            return this
        this = expression
    elif self._match(TokenType.ISNULL):
        this = self.expression(exp.Is, this=this, expression=exp.Null())
    if self._match(TokenType.NOTNULL):
        this = self.expression(exp.Is, this=this, expression=exp.Null())
        this = self.expression(exp.Not, this=this)
    if negate:
        this = self.expression(exp.Not, this=this)
    if self._match(TokenType.IS):
        this = self._parse_is(this)
    return this