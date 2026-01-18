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
def _parse_column_reference(self) -> t.Optional[exp.Expression]:
    this = self._parse_field()
    if not this and self._match(TokenType.VALUES, advance=False) and self.VALUES_FOLLOWED_BY_PAREN and (not self._next or self._next.token_type != TokenType.L_PAREN):
        this = self._parse_id_var()
    return self.expression(exp.Column, this=this) if isinstance(this, exp.Identifier) else this