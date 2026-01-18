from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _parse_date_part(self) -> exp.Expression:
    part = self._parse_type()
    self._match(TokenType.COMMA)
    value = self._parse_bitwise()
    if part and part.is_string:
        part = exp.var(part.name)
    return self.expression(exp.Extract, this=part, expression=value)