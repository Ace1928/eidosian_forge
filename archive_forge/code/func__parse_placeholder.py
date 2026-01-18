from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_placeholder(self) -> t.Optional[exp.Expression]:
    """
            Parse a placeholder expression like SELECT {abc: UInt32} or FROM {table: Identifier}
            https://clickhouse.com/docs/en/sql-reference/syntax#defining-and-using-query-parameters
            """
    if not self._match(TokenType.L_BRACE):
        return None
    this = self._parse_id_var()
    self._match(TokenType.COLON)
    kind = self._parse_types(check_func=False, allow_identifiers=False) or (self._match_text_seq('IDENTIFIER') and 'Identifier')
    if not kind:
        self.raise_error("Expecting a placeholder type or 'Identifier' for tables")
    elif not self._match(TokenType.R_BRACE):
        self.raise_error('Expecting }')
    return self.expression(exp.Placeholder, this=this, kind=kind)