from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _merge_expressions(outer_scope, inner_scope, alias):
    """
    Merge projections of inner query into outer query.

    Args:
        outer_scope (sqlglot.optimizer.scope.Scope)
        inner_scope (sqlglot.optimizer.scope.Scope)
        alias (str)
    """
    outer_columns = defaultdict(list)
    for column in outer_scope.columns:
        if column.table == alias:
            outer_columns[column.name].append(column)
    for expression in inner_scope.expression.expressions:
        projection_name = expression.alias_or_name
        if not projection_name:
            continue
        columns_to_replace = outer_columns.get(projection_name, [])
        expression = expression.unalias()
        must_wrap_expression = not isinstance(expression, SAFE_TO_REPLACE_UNWRAPPED)
        for column in columns_to_replace:
            if isinstance(column.parent, (exp.Unary, exp.Binary)) and must_wrap_expression:
                expression = exp.paren(expression, copy=False)
            column.replace(expression.copy())