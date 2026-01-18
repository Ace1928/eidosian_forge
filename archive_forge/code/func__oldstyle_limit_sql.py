from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _oldstyle_limit_sql(self, expression: exp.Show) -> str:
    limit = self.sql(expression, 'limit')
    offset = self.sql(expression, 'offset')
    if limit:
        limit_offset = f'{offset}, {limit}' if offset else limit
        return f' LIMIT {limit_offset}'
    return ''