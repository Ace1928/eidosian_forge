from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_literal(self, expression: exp.Literal) -> exp.Literal:
    if expression.is_string:
        self._set_type(expression, exp.DataType.Type.VARCHAR)
    elif expression.is_int:
        self._set_type(expression, exp.DataType.Type.INT)
    else:
        self._set_type(expression, exp.DataType.Type.DOUBLE)
    return expression