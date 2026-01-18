from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_set_item_charset(self, kind: str) -> exp.Expression:
    this = self._parse_string() or self._parse_id_var()
    return self.expression(exp.SetItem, this=this, kind=kind)