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
def _parse_create_like(self) -> t.Optional[exp.LikeProperty]:
    table = self._parse_table(schema=True)
    options = []
    while self._match_texts(('INCLUDING', 'EXCLUDING')):
        this = self._prev.text.upper()
        id_var = self._parse_id_var()
        if not id_var:
            return None
        options.append(self.expression(exp.Property, this=this, value=exp.var(id_var.this.upper())))
    return self.expression(exp.LikeProperty, this=table, expressions=options)