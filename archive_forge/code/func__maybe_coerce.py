from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _maybe_coerce(self, type1: exp.DataType | exp.DataType.Type, type2: exp.DataType | exp.DataType.Type) -> exp.DataType | exp.DataType.Type:
    type1_value = type1.this if isinstance(type1, exp.DataType) else type1
    type2_value = type2.this if isinstance(type2, exp.DataType) else type2
    if exp.DataType.Type.NULL in (type1_value, type2_value):
        return exp.DataType.Type.NULL
    if exp.DataType.Type.UNKNOWN in (type1_value, type2_value):
        return exp.DataType.Type.UNKNOWN
    return type2_value if type2_value in self.coerces_to.get(type1_value, {}) else type1_value