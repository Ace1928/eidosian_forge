from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _traverse_ctes(scope):
    sources = {}
    for cte in scope.ctes:
        cte_name = cte.alias
        with_ = scope.expression.args.get('with')
        if with_ and with_.recursive:
            union = cte.this
            if isinstance(union, exp.Union):
                sources[cte_name] = scope.branch(union.this, scope_type=ScopeType.CTE)
        child_scope = None
        for child_scope in _traverse_scope(scope.branch(cte.this, cte_sources=sources, outer_columns=cte.alias_column_names, scope_type=ScopeType.CTE)):
            yield child_scope
        if child_scope:
            sources[cte_name] = child_scope
            scope.cte_scopes.append(child_scope)
    scope.sources.update(sources)
    scope.cte_sources.update(sources)