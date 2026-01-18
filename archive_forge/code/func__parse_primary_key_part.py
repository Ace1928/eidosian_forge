from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_primary_key_part(self) -> t.Optional[exp.Expression]:
    this = self._parse_id_var()
    if not self._match(TokenType.L_PAREN):
        return this
    expression = self._parse_number()
    self._match_r_paren()
    return self.expression(exp.ColumnPrefix, this=this, expression=expression)