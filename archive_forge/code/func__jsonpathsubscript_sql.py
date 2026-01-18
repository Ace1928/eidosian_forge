from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _jsonpathsubscript_sql(self, expression: exp.JSONPathSubscript) -> str:
    this = self.json_path_part(expression.this)
    return str(int(this) + 1) if is_int(this) else this