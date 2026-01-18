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
def _parse_auto_increment(self) -> exp.GeneratedAsIdentityColumnConstraint | exp.AutoIncrementColumnConstraint:
    start = None
    increment = None
    if self._match(TokenType.L_PAREN, advance=False):
        args = self._parse_wrapped_csv(self._parse_bitwise)
        start = seq_get(args, 0)
        increment = seq_get(args, 1)
    elif self._match_text_seq('START'):
        start = self._parse_bitwise()
        self._match_text_seq('INCREMENT')
        increment = self._parse_bitwise()
    if start and increment:
        return exp.GeneratedAsIdentityColumnConstraint(start=start, increment=increment)
    return exp.AutoIncrementColumnConstraint()