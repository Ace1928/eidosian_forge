from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _traverse_scope(scope):
    if isinstance(scope.expression, exp.Select):
        yield from _traverse_select(scope)
    elif isinstance(scope.expression, exp.Union):
        yield from _traverse_ctes(scope)
        yield from _traverse_union(scope)
        return
    elif isinstance(scope.expression, exp.Subquery):
        if scope.is_root:
            yield from _traverse_select(scope)
        else:
            yield from _traverse_subqueries(scope)
    elif isinstance(scope.expression, exp.Table):
        yield from _traverse_tables(scope)
    elif isinstance(scope.expression, exp.UDTF):
        yield from _traverse_udtfs(scope)
    else:
        logger.warning("Cannot traverse scope %s with type '%s'", scope.expression, type(scope.expression))
        return
    yield scope