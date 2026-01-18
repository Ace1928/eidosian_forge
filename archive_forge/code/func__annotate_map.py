from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_map(self, expression):
    self._annotate_args(expression)
    keys = expression.args.get('keys')
    values = expression.args.get('values')
    map_type = exp.DataType(this=exp.DataType.Type.MAP)
    if isinstance(keys, exp.Array) and isinstance(values, exp.Array):
        key_type = seq_get(keys.type.expressions, 0) or exp.DataType.Type.UNKNOWN
        value_type = seq_get(values.type.expressions, 0) or exp.DataType.Type.UNKNOWN
        if key_type != exp.DataType.Type.UNKNOWN and value_type != exp.DataType.Type.UNKNOWN:
            map_type.set('expressions', [key_type, value_type])
            map_type.set('nested', True)
    self._set_type(expression, map_type)
    return expression