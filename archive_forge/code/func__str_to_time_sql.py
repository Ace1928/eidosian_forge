from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _str_to_time_sql(self: Presto.Generator, expression: exp.StrToDate | exp.StrToTime | exp.TsOrDsToDate) -> str:
    return self.func('DATE_PARSE', expression.this, self.format_time(expression))