from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_dot(self, expression: exp.Dot) -> exp.Dot:
    self._annotate_args(expression)
    self._set_type(expression, None)
    this_type = expression.this.type
    if this_type and this_type.is_type(exp.DataType.Type.STRUCT):
        for e in this_type.expressions:
            if e.name == expression.expression.name:
                self._set_type(expression, e.kind)
                break
    return expression