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
def _parse_parameter(self) -> exp.Parameter:
    self._match(TokenType.L_BRACE)
    this = self._parse_identifier() or self._parse_primary_or_var()
    expression = self._match(TokenType.COLON) and (self._parse_identifier() or self._parse_primary_or_var())
    self._match(TokenType.R_BRACE)
    return self.expression(exp.Parameter, this=this, expression=expression)