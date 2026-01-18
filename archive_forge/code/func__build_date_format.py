from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _build_date_format(args: t.List) -> exp.TimeToStr:
    expr = build_formatted_time(exp.TimeToStr, 'clickhouse')(args)
    timezone = seq_get(args, 2)
    if timezone:
        expr.set('timezone', timezone)
    return expr