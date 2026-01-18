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
def _parse_predict(self) -> exp.Predict:
    self._match_text_seq('MODEL')
    this = self._parse_table()
    self._match(TokenType.COMMA)
    self._match_text_seq('TABLE')
    return self.expression(exp.Predict, this=this, expression=self._parse_table(), params_struct=self._match(TokenType.COMMA) and self._parse_bitwise())