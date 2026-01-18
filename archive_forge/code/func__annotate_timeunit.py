from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_timeunit(self, expression: exp.TimeUnit | exp.DateTrunc) -> exp.TimeUnit | exp.DateTrunc:
    self._annotate_args(expression)
    if expression.this.type.this in exp.DataType.TEXT_TYPES:
        datatype = _coerce_date_literal(expression.this, expression.unit)
    elif expression.this.type.this in exp.DataType.TEMPORAL_TYPES:
        datatype = _coerce_date(expression.this, expression.unit)
    else:
        datatype = exp.DataType.Type.UNKNOWN
    self._set_type(expression, datatype)
    return expression