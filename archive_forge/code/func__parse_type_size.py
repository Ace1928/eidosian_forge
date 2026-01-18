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
def _parse_type_size(self) -> t.Optional[exp.DataTypeParam]:
    this = self._parse_type()
    if not this:
        return None
    if isinstance(this, exp.Column) and (not this.table):
        this = exp.var(this.name.upper())
    return self.expression(exp.DataTypeParam, this=this, expression=self._parse_var(any_token=True))