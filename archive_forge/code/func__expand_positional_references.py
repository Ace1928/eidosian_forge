from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def _expand_positional_references(scope: Scope, expressions: t.Iterable[exp.Expression], alias: bool=False) -> t.List[exp.Expression]:
    new_nodes: t.List[exp.Expression] = []
    for node in expressions:
        if node.is_int:
            select = _select_by_pos(scope, t.cast(exp.Literal, node))
            if alias:
                new_nodes.append(exp.column(select.args['alias'].copy()))
            else:
                select = select.this
                if isinstance(select, exp.CONSTANTS) or select.find(exp.Explode, exp.Unnest):
                    new_nodes.append(node)
                else:
                    new_nodes.append(select.copy())
        else:
            new_nodes.append(node)
    return new_nodes