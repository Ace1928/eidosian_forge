from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _property_sql(self: Hive.Generator, expression: exp.Property) -> str:
    return f'{self.property_name(expression, string_key=True)}={self.sql(expression, 'value')}'