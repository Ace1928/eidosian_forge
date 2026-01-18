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
def _parse_json_key_value(self) -> t.Optional[exp.JSONKeyValue]:
    self._match_text_seq('KEY')
    key = self._parse_column()
    self._match_set(self.JSON_KEY_VALUE_SEPARATOR_TOKENS)
    self._match_text_seq('VALUE')
    value = self._parse_bitwise()
    if not key and (not value):
        return None
    return self.expression(exp.JSONKeyValue, this=key, expression=value)