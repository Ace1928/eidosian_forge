from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def _parse_rangen(self):
    this = self._parse_id_var()
    self._match(TokenType.BETWEEN)
    expressions = self._parse_csv(self._parse_conjunction)
    each = self._match_text_seq('EACH') and self._parse_conjunction()
    return self.expression(exp.RangeN, this=this, expressions=expressions, each=each)