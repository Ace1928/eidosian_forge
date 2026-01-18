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
def _parse_volatile_property(self) -> exp.VolatileProperty | exp.StabilityProperty:
    if self._index >= 2:
        pre_volatile_token = self._tokens[self._index - 2]
    else:
        pre_volatile_token = None
    if pre_volatile_token and pre_volatile_token.token_type in self.PRE_VOLATILE_TOKENS:
        return exp.VolatileProperty()
    return self.expression(exp.StabilityProperty, this=exp.Literal.string('VOLATILE'))