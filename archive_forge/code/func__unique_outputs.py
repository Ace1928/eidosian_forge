from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope
def _unique_outputs(scope):
    """Determine output columns of `scope` that must have a unique combination per row"""
    if scope.expression.args.get('distinct'):
        return set(scope.expression.named_selects)
    group = scope.expression.args.get('group')
    if group:
        grouped_expressions = set(group.expressions)
        grouped_outputs = set()
        unique_outputs = set()
        for select in scope.expression.selects:
            output = select.unalias()
            if output in grouped_expressions:
                grouped_outputs.add(output)
                unique_outputs.add(select.alias_or_name)
        if not grouped_expressions.difference(grouped_outputs):
            return unique_outputs
        else:
            return set()
    if _has_single_output_row(scope):
        return set(scope.expression.named_selects)
    return set()