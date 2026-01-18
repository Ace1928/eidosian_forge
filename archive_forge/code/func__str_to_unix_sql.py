from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _str_to_unix_sql(self: Hive.Generator, expression: exp.StrToUnix) -> str:
    return self.func('UNIX_TIMESTAMP', expression.this, time_format('hive')(self, expression))