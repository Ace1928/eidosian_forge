from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_wrapped_id_vars(self, optional: bool=False) -> t.List[exp.Expression]:
    return super()._parse_wrapped_id_vars(optional=True)