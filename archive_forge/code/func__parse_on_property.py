from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_on_property(self) -> t.Optional[exp.Expression]:
    index = self._index
    if self._match_text_seq('CLUSTER'):
        this = self._parse_id_var()
        if this:
            return self.expression(exp.OnCluster, this=this)
        else:
            self._retreat(index)
    return None