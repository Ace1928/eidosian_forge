from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_div(self, expression: exp.Div) -> exp.Div:
    self._annotate_args(expression)
    left_type, right_type = (expression.left.type.this, expression.right.type.this)
    if expression.args.get('typed') and left_type in exp.DataType.INTEGER_TYPES and (right_type in exp.DataType.INTEGER_TYPES):
        self._set_type(expression, exp.DataType.Type.BIGINT)
    else:
        self._set_type(expression, self._maybe_coerce(left_type, right_type))
        if expression.type and expression.type.this not in exp.DataType.REAL_TYPES:
            self._set_type(expression, self._maybe_coerce(expression.type, exp.DataType.Type.DOUBLE))
    return expression