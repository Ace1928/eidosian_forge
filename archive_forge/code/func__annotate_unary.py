from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_unary(self, expression: E) -> E:
    self._annotate_args(expression)
    if isinstance(expression, exp.Condition) and (not isinstance(expression, exp.Paren)):
        self._set_type(expression, exp.DataType.Type.BOOLEAN)
    else:
        self._set_type(expression, expression.this.type)
    return expression