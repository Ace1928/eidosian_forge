from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _trim_sql(self: MySQL.Generator, expression: exp.Trim) -> str:
    target = self.sql(expression, 'this')
    trim_type = self.sql(expression, 'position')
    remove_chars = self.sql(expression, 'expression')
    if not remove_chars:
        return self.trim_sql(expression)
    trim_type = f'{trim_type} ' if trim_type else ''
    remove_chars = f'{remove_chars} ' if remove_chars else ''
    from_part = 'FROM ' if trim_type or remove_chars else ''
    return f'TRIM({trim_type}{remove_chars}{from_part}{target})'