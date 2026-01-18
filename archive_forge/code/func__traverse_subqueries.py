from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _traverse_subqueries(scope):
    for subquery in scope.subqueries:
        top = None
        for child_scope in _traverse_scope(scope.branch(subquery, scope_type=ScopeType.SUBQUERY)):
            yield child_scope
            top = child_scope
        scope.subquery_scopes.append(top)