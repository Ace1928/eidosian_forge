from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _initcap_sql(self: Presto.Generator, expression: exp.Initcap) -> str:
    regex = '(\\w)(\\w*)'
    return f"REGEXP_REPLACE({self.sql(expression, 'this')}, '{regex}', x -> UPPER(x[1]) || LOWER(x[2]))"