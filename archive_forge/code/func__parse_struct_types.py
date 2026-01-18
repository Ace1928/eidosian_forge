from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _parse_struct_types(self, type_required: bool=False) -> t.Optional[exp.Expression]:
    return self._parse_field_def()