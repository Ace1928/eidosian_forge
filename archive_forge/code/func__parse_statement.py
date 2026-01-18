from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_statement(self) -> t.Optional[exp.Expression]:
    expression = self._parse_expression()
    expression = expression if expression else self._parse_query()
    return expression