from collections import defaultdict
from sqlglot import alias, exp
from sqlglot.optimizer.qualify_columns import Resolver
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import ensure_schema
def _remove_unused_selections(scope, parent_selections, schema, alias_count):
    order = scope.expression.args.get('order')
    if order:
        order_refs = {c.name for c in order.find_all(exp.Column) if not c.table}
    else:
        order_refs = set()
    new_selections = []
    removed = False
    star = False
    is_agg = False
    select_all = SELECT_ALL in parent_selections
    for selection in scope.expression.selects:
        name = selection.alias_or_name
        if select_all or name in parent_selections or name in order_refs or (alias_count > 0):
            new_selections.append(selection)
            alias_count -= 1
        else:
            if selection.is_star:
                star = True
            removed = True
        if not is_agg and selection.find(exp.AggFunc):
            is_agg = True
    if star:
        resolver = Resolver(scope, schema)
        names = {s.alias_or_name for s in new_selections}
        for name in sorted(parent_selections):
            if name not in names:
                new_selections.append(alias(exp.column(name, table=resolver.get_table(name)), name, copy=False))
    if not new_selections:
        new_selections.append(default_selection(is_agg))
    scope.expression.select(*new_selections, append=False, copy=False)
    if removed:
        scope.clear_cache()