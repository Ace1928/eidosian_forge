from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _substring_sql(self: Postgres.Generator, expression: exp.Substring) -> str:
    this = self.sql(expression, 'this')
    start = self.sql(expression, 'start')
    length = self.sql(expression, 'length')
    from_part = f' FROM {start}' if start else ''
    for_part = f' FOR {length}' if length else ''
    return f'SUBSTRING({this}{from_part}{for_part})'