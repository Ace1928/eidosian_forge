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
def _parse_dict_property(self, this: str) -> exp.DictProperty:
    settings = []
    self._match_l_paren()
    kind = self._parse_id_var()
    if self._match(TokenType.L_PAREN):
        while True:
            key = self._parse_id_var()
            value = self._parse_primary()
            if not key and value is None:
                break
            settings.append(self.expression(exp.DictSubProperty, this=key, value=value))
        self._match(TokenType.R_PAREN)
    self._match_r_paren()
    return self.expression(exp.DictProperty, this=this, kind=kind.this if kind else None, settings=settings)