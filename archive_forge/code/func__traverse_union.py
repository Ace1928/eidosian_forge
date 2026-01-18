from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _traverse_union(scope):
    prev_scope = None
    union_scope_stack = [scope]
    expression_stack = [scope.expression.right, scope.expression.left]
    while expression_stack:
        expression = expression_stack.pop()
        union_scope = union_scope_stack[-1]
        new_scope = union_scope.branch(expression, outer_columns=union_scope.outer_columns, scope_type=ScopeType.UNION)
        if isinstance(expression, exp.Union):
            yield from _traverse_ctes(new_scope)
            union_scope_stack.append(new_scope)
            expression_stack.extend([expression.right, expression.left])
            continue
        for scope in _traverse_scope(new_scope):
            yield scope
        if prev_scope:
            union_scope_stack.pop()
            union_scope.union_scopes = [prev_scope, scope]
            prev_scope = union_scope
            yield union_scope
        else:
            prev_scope = scope