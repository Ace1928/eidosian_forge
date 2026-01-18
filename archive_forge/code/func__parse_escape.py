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
def _parse_escape(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    if not self._match(TokenType.ESCAPE):
        return this
    return self.expression(exp.Escape, this=this, expression=self._parse_string())