from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _to_date_sql(self: Hive.Generator, expression: exp.TsOrDsToDate) -> str:
    time_format = self.format_time(expression)
    if time_format and time_format not in (Hive.TIME_FORMAT, Hive.DATE_FORMAT):
        return self.func('TO_DATE', expression.this, time_format)
    if isinstance(expression.this, exp.TsOrDsToDate):
        return self.sql(expression, 'this')
    return self.func('TO_DATE', expression.this)