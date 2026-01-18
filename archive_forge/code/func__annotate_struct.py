from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_struct(self, expression: exp.Struct) -> exp.Struct:
    self._annotate_args(expression)
    self._set_type(expression, exp.DataType(this=exp.DataType.Type.STRUCT, expressions=[self._annotate_struct_value(expr) for expr in expression.expressions], nested=True))
    return expression