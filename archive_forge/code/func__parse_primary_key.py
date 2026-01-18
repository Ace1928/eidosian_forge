from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_primary_key(self, wrapped_optional: bool=False, in_props: bool=False) -> exp.PrimaryKeyColumnConstraint | exp.PrimaryKey:
    return super()._parse_primary_key(wrapped_optional=wrapped_optional or in_props, in_props=in_props)