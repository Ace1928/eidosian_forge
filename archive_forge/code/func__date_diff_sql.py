from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _date_diff_sql(self: Postgres.Generator, expression: exp.DateDiff) -> str:
    unit = expression.text('unit').upper()
    factor = DATE_DIFF_FACTOR.get(unit)
    end = f'CAST({self.sql(expression, 'this')} AS TIMESTAMP)'
    start = f'CAST({self.sql(expression, 'expression')} AS TIMESTAMP)'
    if factor is not None:
        return f'CAST(EXTRACT(epoch FROM {end} - {start}){factor} AS BIGINT)'
    age = f'AGE({end}, {start})'
    if unit == 'WEEK':
        unit = f'EXTRACT(days FROM ({end} - {start})) / 7'
    elif unit == 'MONTH':
        unit = f'EXTRACT(year FROM {age}) * 12 + EXTRACT(month FROM {age})'
    elif unit == 'QUARTER':
        unit = f'EXTRACT(year FROM {age}) * 4 + EXTRACT(month FROM {age}) / 3'
    elif unit == 'YEAR':
        unit = f'EXTRACT(year FROM {age})'
    else:
        unit = age
    return f'CAST({unit} AS BIGINT)'