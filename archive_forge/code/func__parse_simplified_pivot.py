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
def _parse_simplified_pivot(self) -> exp.Pivot:

    def _parse_on() -> t.Optional[exp.Expression]:
        this = self._parse_bitwise()
        return self._parse_in(this) if self._match(TokenType.IN) else this
    this = self._parse_table()
    expressions = self._match(TokenType.ON) and self._parse_csv(_parse_on)
    using = self._match(TokenType.USING) and self._parse_csv(lambda: self._parse_alias(self._parse_function()))
    group = self._parse_group()
    return self.expression(exp.Pivot, this=this, expressions=expressions, using=using, group=group)