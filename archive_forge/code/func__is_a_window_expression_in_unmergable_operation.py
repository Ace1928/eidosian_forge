from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _is_a_window_expression_in_unmergable_operation():
    window_expressions = inner_select.find_all(exp.Window)
    window_alias_names = {window.parent.alias_or_name for window in window_expressions}
    inner_select_name = from_or_join.alias_or_name
    unmergable_window_columns = [column for column in outer_scope.columns if column.find_ancestor(exp.Where, exp.Group, exp.Order, exp.Join, exp.Having, exp.AggFunc)]
    window_expressions_in_unmergable = [column for column in unmergable_window_columns if column.table == inner_select_name and column.name in window_alias_names]
    return any(window_expressions_in_unmergable)