from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _jsonpathkey_sql(self, expression: exp.JSONPathKey) -> str:
    if isinstance(expression.this, exp.JSONPathWildcard):
        self.unsupported('Unsupported wildcard in JSONPathKey expression')
        return ''
    return super()._jsonpathkey_sql(expression)