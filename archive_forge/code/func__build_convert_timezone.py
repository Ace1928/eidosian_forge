from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _build_convert_timezone(args: t.List) -> t.Union[exp.Anonymous, exp.AtTimeZone]:
    if len(args) == 3:
        return exp.Anonymous(this='CONVERT_TIMEZONE', expressions=args)
    return exp.AtTimeZone(this=seq_get(args, 1), zone=seq_get(args, 0))