from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _str_to_date_sql(self: MySQL.Generator, expression: exp.StrToDate | exp.StrToTime | exp.TsOrDsToDate) -> str:
    return self.func('STR_TO_DATE', expression.this, self.format_time(expression))