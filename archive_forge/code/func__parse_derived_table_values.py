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
def _parse_derived_table_values(self) -> t.Optional[exp.Values]:
    is_derived = self._match_pair(TokenType.L_PAREN, TokenType.VALUES)
    if not is_derived and (not self._match_text_seq('VALUES')):
        return None
    expressions = self._parse_csv(self._parse_value)
    alias = self._parse_table_alias()
    if is_derived:
        self._match_r_paren()
    return self.expression(exp.Values, expressions=expressions, alias=alias or self._parse_table_alias())