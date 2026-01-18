from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _pivot_column_names(self, aggregations: t.List[exp.Expression]) -> t.List[str]:
    if len(aggregations) == 1:
        return super()._pivot_column_names(aggregations)
    return pivot_column_names(aggregations, dialect='duckdb')