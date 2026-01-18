import itertools
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import build_scope
def _eliminate_cte(scope, existing_ctes, taken):
    parent = scope.expression.parent
    name, cte = _new_cte(scope, existing_ctes, taken)
    with_ = parent.parent
    parent.pop()
    if not with_.expressions:
        with_.pop()
    for child_scope in scope.parent.traverse():
        for table, source in child_scope.selected_sources.values():
            if source is scope:
                new_table = exp.alias_(exp.table_(name), alias=table.alias_or_name, copy=False)
                table.replace(new_table)
    return cte