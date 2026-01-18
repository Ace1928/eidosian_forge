from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_struct_value(self, expression: exp.Expression) -> t.Optional[exp.DataType] | exp.ColumnDef:
    alias = expression.args.get('alias')
    if alias:
        return exp.ColumnDef(this=alias.copy(), kind=expression.type)
    if expression.expression:
        return exp.ColumnDef(this=expression.this.copy(), kind=expression.expression.type)
    return expression.type