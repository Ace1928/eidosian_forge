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
def _expand_order_by(scope: Scope, resolver: Resolver) -> None:
    order = scope.expression.args.get('order')
    if not order:
        return
    ordereds = order.expressions
    for ordered, new_expression in zip(ordereds, _expand_positional_references(scope, (o.this for o in ordereds), alias=True)):
        for agg in ordered.find_all(exp.AggFunc):
            for col in agg.find_all(exp.Column):
                if not col.table:
                    col.set('table', resolver.get_table(col.name))
        ordered.set('this', new_expression)
    if scope.expression.args.get('group'):
        selects = {s.this: exp.column(s.alias_or_name) for s in scope.expression.selects}
        for ordered in ordereds:
            ordered = ordered.this
            ordered.replace(exp.to_identifier(_select_by_pos(scope, ordered).alias) if ordered.is_int else selects.get(ordered, ordered))