from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_binary(self, expression: B) -> B:
    self._annotate_args(expression)
    left, right = (expression.left, expression.right)
    left_type, right_type = (left.type.this, right.type.this)
    if isinstance(expression, exp.Connector):
        if left_type == exp.DataType.Type.NULL and right_type == exp.DataType.Type.NULL:
            self._set_type(expression, exp.DataType.Type.NULL)
        elif exp.DataType.Type.NULL in (left_type, right_type):
            self._set_type(expression, exp.DataType.build('NULLABLE', expressions=exp.DataType.build('BOOLEAN')))
        else:
            self._set_type(expression, exp.DataType.Type.BOOLEAN)
    elif isinstance(expression, exp.Predicate):
        self._set_type(expression, exp.DataType.Type.BOOLEAN)
    elif (left_type, right_type) in self.binary_coercions:
        self._set_type(expression, self.binary_coercions[left_type, right_type](left, right))
    else:
        self._set_type(expression, self._maybe_coerce(left_type, right_type))
    return expression