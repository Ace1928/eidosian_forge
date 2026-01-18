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
def _parse_checksum(self) -> exp.ChecksumProperty:
    self._match(TokenType.EQ)
    on = None
    if self._match(TokenType.ON):
        on = True
    elif self._match_text_seq('OFF'):
        on = False
    return self.expression(exp.ChecksumProperty, on=on, default=self._match(TokenType.DEFAULT))