from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_args(self, expression: E) -> E:
    for value in expression.iter_expressions():
        self._maybe_annotate(value)
    return expression