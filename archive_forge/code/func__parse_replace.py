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
def _parse_replace(self) -> t.Optional[t.List[exp.Expression]]:
    if not self._match(TokenType.REPLACE):
        return None
    if self._match(TokenType.L_PAREN, advance=False):
        return self._parse_wrapped_csv(self._parse_expression)
    replace_expression = self._parse_expression()
    return [replace_expression] if replace_expression else None