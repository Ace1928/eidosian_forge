from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _date_delta_sql(self: DuckDB.Generator, expression: exp.DateAdd | exp.DateSub | exp.TimeAdd) -> str:
    this = self.sql(expression, 'this')
    unit = unit_to_var(expression)
    op = '+' if isinstance(expression, (exp.DateAdd, exp.TimeAdd)) else '-'
    return f'{this} {op} {self.sql(exp.Interval(this=expression.expression, unit=unit))}'