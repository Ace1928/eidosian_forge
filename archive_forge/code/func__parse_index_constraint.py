from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_index_constraint(self, kind: t.Optional[str]=None) -> exp.IndexColumnConstraint:
    this = self._parse_id_var()
    expression = self._parse_conjunction()
    index_type = self._match_text_seq('TYPE') and (self._parse_function() or self._parse_var())
    granularity = self._match_text_seq('GRANULARITY') and self._parse_term()
    return self.expression(exp.IndexColumnConstraint, this=this, expression=expression, index_type=index_type, granularity=granularity)