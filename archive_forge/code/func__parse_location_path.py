from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_location_path(self) -> exp.Var:
    parts = [self._advance_any(ignore_reserved=True)]
    while self._is_connected() and (not self._match(TokenType.COMMA, advance=False)):
        parts.append(self._advance_any(ignore_reserved=True))
    return exp.var(''.join((part.text for part in parts if part)))