from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _parse_map(self) -> exp.ToMap | exp.Map:
    if self._match(TokenType.L_BRACE, advance=False):
        return self.expression(exp.ToMap, this=self._parse_bracket())
    args = self._parse_wrapped_csv(self._parse_conjunction)
    return self.expression(exp.Map, keys=seq_get(args, 0), values=seq_get(args, 1))