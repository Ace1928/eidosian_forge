from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _build_date_time_add(expr_type: t.Type[E]) -> t.Callable[[t.List], E]:

    def _builder(args: t.List) -> E:
        return expr_type(this=seq_get(args, 2), expression=seq_get(args, 1), unit=_map_date_part(seq_get(args, 0)))
    return _builder