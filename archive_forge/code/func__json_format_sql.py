from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _json_format_sql(self: DuckDB.Generator, expression: exp.JSONFormat) -> str:
    sql = self.func('TO_JSON', expression.this, expression.args.get('options'))
    return f'CAST({sql} AS TEXT)'