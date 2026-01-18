from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _build_to_char(args: t.List) -> exp.TimeToStr:
    fmt = seq_get(args, 1)
    if isinstance(fmt, exp.Literal):
        fmt.set('this', fmt.this.upper())
    return build_formatted_time(exp.TimeToStr, 'teradata')(args)