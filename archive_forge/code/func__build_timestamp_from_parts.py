from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _build_timestamp_from_parts(args: t.List) -> exp.Func:
    if len(args) == 2:
        return exp.Anonymous(this='TIMESTAMP_FROM_PARTS', expressions=args)
    return exp.TimestampFromParts.from_arg_list(args)