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
def _parse_into(self) -> t.Optional[exp.Into]:
    if not self._match(TokenType.INTO):
        return None
    temp = self._match(TokenType.TEMPORARY)
    unlogged = self._match_text_seq('UNLOGGED')
    self._match(TokenType.TABLE)
    return self.expression(exp.Into, this=self._parse_table(schema=True), temporary=temp, unlogged=unlogged)