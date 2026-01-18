from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _merge_order(outer_scope, inner_scope):
    """
    Merge ORDER clause of inner query into outer query.

    Args:
        outer_scope (sqlglot.optimizer.scope.Scope)
        inner_scope (sqlglot.optimizer.scope.Scope)
    """
    if any((outer_scope.expression.args.get(arg) for arg in ['group', 'distinct', 'having', 'order'])) or len(outer_scope.selected_sources) != 1 or any((expression.find(exp.AggFunc) for expression in outer_scope.expression.expressions)):
        return
    outer_scope.expression.set('order', inner_scope.expression.args.get('order'))