from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _date_sql(self: DuckDB.Generator, expression: exp.Date) -> str:
    result = f'CAST({self.sql(expression, 'this')} AS DATE)'
    zone = self.sql(expression, 'zone')
    if zone:
        date_str = self.func('STRFTIME', result, "'%d/%m/%Y'")
        date_str = f"{date_str} || ' ' || {zone}"
        result = self.func('STRPTIME', date_str, "'%d/%m/%Y %Z'")
    return result