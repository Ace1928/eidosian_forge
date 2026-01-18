from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_cte(self) -> exp.CTE:
    cte: t.Optional[exp.CTE] = self._try_parse(super()._parse_cte)
    if not cte:
        cte = self.expression(exp.CTE, this=self._parse_conjunction(), alias=self._parse_table_alias(), scalar=True)
    return cte