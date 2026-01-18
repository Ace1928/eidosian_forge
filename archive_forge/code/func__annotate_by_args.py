from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
@t.no_type_check
def _annotate_by_args(self, expression: E, *args: str, promote: bool=False, array: bool=False) -> E:
    self._annotate_args(expression)
    expressions: t.List[exp.Expression] = []
    for arg in args:
        arg_expr = expression.args.get(arg)
        expressions.extend((expr for expr in ensure_list(arg_expr) if expr))
    last_datatype = None
    for expr in expressions:
        expr_type = expr.type
        if expr_type.args.get('nested'):
            last_datatype = expr_type
            break
        if not expr_type.is_type(exp.DataType.Type.NULL, exp.DataType.Type.UNKNOWN):
            last_datatype = self._maybe_coerce(last_datatype or expr_type, expr_type)
    self._set_type(expression, last_datatype or exp.DataType.Type.UNKNOWN)
    if promote:
        if expression.type.this in exp.DataType.INTEGER_TYPES:
            self._set_type(expression, exp.DataType.Type.BIGINT)
        elif expression.type.this in exp.DataType.FLOAT_TYPES:
            self._set_type(expression, exp.DataType.Type.DOUBLE)
    if array:
        self._set_type(expression, exp.DataType(this=exp.DataType.Type.ARRAY, expressions=[expression.type], nested=True))
    return expression