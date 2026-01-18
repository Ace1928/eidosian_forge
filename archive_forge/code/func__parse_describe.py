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
def _parse_describe(self) -> exp.Describe:
    kind = self._match_set(self.CREATABLES) and self._prev.text
    style = self._match_texts(('EXTENDED', 'FORMATTED', 'HISTORY')) and self._prev.text.upper()
    if not self._match_set(self.ID_VAR_TOKENS, advance=False):
        style = None
        self._retreat(self._index - 1)
    this = self._parse_table(schema=True)
    properties = self._parse_properties()
    expressions = properties.expressions if properties else None
    return self.expression(exp.Describe, this=this, style=style, kind=kind, expressions=expressions)