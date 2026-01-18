import itertools
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import build_scope
def _eliminate_derived_table(scope, existing_ctes, taken):
    if scope.parent.pivots or isinstance(scope.parent.expression, exp.Lateral):
        return None
    to_replace = scope.expression.parent.unwrap()
    name, cte = _new_cte(scope, existing_ctes, taken)
    table = exp.alias_(exp.table_(name), alias=to_replace.alias or name)
    table.set('joins', to_replace.args.get('joins'))
    to_replace.replace(table)
    return cte