from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_explode(self, expression: exp.Explode) -> exp.Explode:
    self._annotate_args(expression)
    self._set_type(expression, seq_get(expression.this.type.expressions, 0))
    return expression