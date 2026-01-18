from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _flatten_structured_type(expression: exp.DataType) -> exp.DataType:
    if expression.this in exp.DataType.NESTED_TYPES:
        expression.set('expressions', None)
    return expression