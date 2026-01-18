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
def _parse_foreign_key(self) -> exp.ForeignKey:
    expressions = self._parse_wrapped_id_vars()
    reference = self._parse_references()
    options = {}
    while self._match(TokenType.ON):
        if not self._match_set((TokenType.DELETE, TokenType.UPDATE)):
            self.raise_error('Expected DELETE or UPDATE')
        kind = self._prev.text.lower()
        if self._match_text_seq('NO', 'ACTION'):
            action = 'NO ACTION'
        elif self._match(TokenType.SET):
            self._match_set((TokenType.NULL, TokenType.DEFAULT))
            action = 'SET ' + self._prev.text.upper()
        else:
            self._advance()
            action = self._prev.text.upper()
        options[kind] = action
    return self.expression(exp.ForeignKey, expressions=expressions, reference=reference, **options)