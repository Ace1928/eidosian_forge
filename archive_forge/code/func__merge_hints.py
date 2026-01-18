from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _merge_hints(outer_scope, inner_scope):
    inner_scope_hint = inner_scope.expression.args.get('hint')
    if not inner_scope_hint:
        return
    outer_scope_hint = outer_scope.expression.args.get('hint')
    if outer_scope_hint:
        for hint_expression in inner_scope_hint.expressions:
            outer_scope_hint.append('expressions', hint_expression)
    else:
        outer_scope.expression.set('hint', inner_scope_hint)