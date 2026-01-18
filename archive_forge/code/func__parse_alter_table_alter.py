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
def _parse_alter_table_alter(self) -> exp.AlterColumn:
    self._match(TokenType.COLUMN)
    column = self._parse_field(any_token=True)
    if self._match_pair(TokenType.DROP, TokenType.DEFAULT):
        return self.expression(exp.AlterColumn, this=column, drop=True)
    if self._match_pair(TokenType.SET, TokenType.DEFAULT):
        return self.expression(exp.AlterColumn, this=column, default=self._parse_conjunction())
    if self._match(TokenType.COMMENT):
        return self.expression(exp.AlterColumn, this=column, comment=self._parse_string())
    self._match_text_seq('SET', 'DATA')
    self._match_text_seq('TYPE')
    return self.expression(exp.AlterColumn, this=column, dtype=self._parse_types(), collate=self._match(TokenType.COLLATE) and self._parse_term(), using=self._match(TokenType.USING) and self._parse_conjunction())