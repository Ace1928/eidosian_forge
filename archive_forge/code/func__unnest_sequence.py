from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _unnest_sequence(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Table):
        if isinstance(expression.this, exp.GenerateSeries):
            unnest = exp.Unnest(expressions=[expression.this])
            if expression.alias:
                return exp.alias_(unnest, alias='_u', table=[expression.alias], copy=False)
            return unnest
    return expression