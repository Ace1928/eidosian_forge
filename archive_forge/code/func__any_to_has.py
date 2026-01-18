from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _any_to_has(self, expression: exp.EQ | exp.NEQ, default: t.Callable[[t.Any], str], prefix: str='') -> str:
    if isinstance(expression.left, exp.Any):
        arr = expression.left
        this = expression.right
    elif isinstance(expression.right, exp.Any):
        arr = expression.right
        this = expression.left
    else:
        return default(expression)
    return prefix + self.func('has', arr.this.unnest(), this)