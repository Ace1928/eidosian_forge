from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _regexpilike_sql(self: Snowflake.Generator, expression: exp.RegexpILike) -> str:
    flag = expression.text('flag')
    if 'i' not in flag:
        flag += 'i'
    return self.func('REGEXP_LIKE', expression.this, expression.expression, exp.Literal.string(flag))