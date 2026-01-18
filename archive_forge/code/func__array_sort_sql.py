from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _array_sort_sql(self: DuckDB.Generator, expression: exp.ArraySort) -> str:
    if expression.expression:
        self.unsupported('DuckDB ARRAY_SORT does not support a comparator')
    return self.func('ARRAY_SORT', expression.this)