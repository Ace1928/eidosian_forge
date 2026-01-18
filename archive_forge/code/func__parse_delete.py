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
def _parse_delete(self) -> exp.Delete:
    tables = None
    comments = self._prev_comments
    if not self._match(TokenType.FROM, advance=False):
        tables = self._parse_csv(self._parse_table) or None
    returning = self._parse_returning()
    return self.expression(exp.Delete, comments=comments, tables=tables, this=self._match(TokenType.FROM) and self._parse_table(joins=True), using=self._match(TokenType.USING) and self._parse_table(joins=True), where=self._parse_where(), returning=returning or self._parse_returning(), limit=self._parse_limit())