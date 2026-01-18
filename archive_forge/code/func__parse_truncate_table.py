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
def _parse_truncate_table(self) -> t.Optional[exp.TruncateTable] | exp.Expression:
    start = self._prev
    if self._match(TokenType.L_PAREN):
        self._retreat(self._index - 2)
        return self._parse_function()
    is_database = self._match(TokenType.DATABASE)
    self._match(TokenType.TABLE)
    exists = self._parse_exists(not_=False)
    expressions = self._parse_csv(lambda: self._parse_table(schema=True, is_db_reference=is_database))
    cluster = self._parse_on_property() if self._match(TokenType.ON) else None
    if self._match_text_seq('RESTART', 'IDENTITY'):
        identity = 'RESTART'
    elif self._match_text_seq('CONTINUE', 'IDENTITY'):
        identity = 'CONTINUE'
    else:
        identity = None
    if self._match_text_seq('CASCADE') or self._match_text_seq('RESTRICT'):
        option = self._prev.text
    else:
        option = None
    partition = self._parse_partition()
    if self._curr:
        return self._parse_as_command(start)
    return self.expression(exp.TruncateTable, expressions=expressions, is_database=is_database, exists=exists, cluster=cluster, identity=identity, option=option, partition=partition)