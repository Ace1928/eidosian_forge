from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_alter_table_swap(self) -> exp.SwapTable:
    self._match_text_seq('WITH')
    return self.expression(exp.SwapTable, this=self._parse_table(schema=True))