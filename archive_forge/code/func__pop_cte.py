from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _pop_cte(inner_scope):
    """
    Remove CTE from the AST.

    Args:
        inner_scope (sqlglot.optimizer.scope.Scope)
    """
    cte = inner_scope.expression.parent
    with_ = cte.parent
    if len(with_.expressions) == 1:
        with_.pop()
    else:
        cte.pop()