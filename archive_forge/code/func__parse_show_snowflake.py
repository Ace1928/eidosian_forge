from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_show_snowflake(self, this: str) -> exp.Show:
    scope = None
    scope_kind = None
    terse = self._tokens[self._index - 2].text.upper() == 'TERSE'
    history = self._match_text_seq('HISTORY')
    like = self._parse_string() if self._match(TokenType.LIKE) else None
    if self._match(TokenType.IN):
        if self._match_text_seq('ACCOUNT'):
            scope_kind = 'ACCOUNT'
        elif self._match_set(self.DB_CREATABLES):
            scope_kind = self._prev.text.upper()
            if self._curr:
                scope = self._parse_table_parts()
        elif self._curr:
            scope_kind = 'SCHEMA' if this in self.SCHEMA_KINDS else 'TABLE'
            scope = self._parse_table_parts()
    return self.expression(exp.Show, **{'terse': terse, 'this': this, 'history': history, 'like': like, 'scope': scope, 'scope_kind': scope_kind, 'starts_with': self._match_text_seq('STARTS', 'WITH') and self._parse_string(), 'limit': self._parse_limit(), 'from': self._parse_string() if self._match(TokenType.FROM) else None})