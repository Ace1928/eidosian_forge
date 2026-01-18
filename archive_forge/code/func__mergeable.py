from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _mergeable(outer_scope, inner_scope, leave_tables_isolated, from_or_join):
    """
    Return True if `inner_select` can be merged into outer query.

    Args:
        outer_scope (Scope)
        inner_scope (Scope)
        leave_tables_isolated (bool)
        from_or_join (exp.From|exp.Join)
    Returns:
        bool: True if can be merged
    """
    inner_select = inner_scope.expression.unnest()

    def _is_a_window_expression_in_unmergable_operation():
        window_expressions = inner_select.find_all(exp.Window)
        window_alias_names = {window.parent.alias_or_name for window in window_expressions}
        inner_select_name = from_or_join.alias_or_name
        unmergable_window_columns = [column for column in outer_scope.columns if column.find_ancestor(exp.Where, exp.Group, exp.Order, exp.Join, exp.Having, exp.AggFunc)]
        window_expressions_in_unmergable = [column for column in unmergable_window_columns if column.table == inner_select_name and column.name in window_alias_names]
        return any(window_expressions_in_unmergable)

    def _outer_select_joins_on_inner_select_join():
        """
        All columns from the inner select in the ON clause must be from the first FROM table.

        That is, this can be merged:
            SELECT * FROM x JOIN (SELECT y.a AS a FROM y JOIN z) AS q ON x.a = q.a
                                         ^^^           ^
        But this can't:
            SELECT * FROM x JOIN (SELECT z.a AS a FROM y JOIN z) AS q ON x.a = q.a
                                         ^^^                  ^
        """
        if not isinstance(from_or_join, exp.Join):
            return False
        alias = from_or_join.alias_or_name
        on = from_or_join.args.get('on')
        if not on:
            return False
        selections = [c.name for c in on.find_all(exp.Column) if c.table == alias]
        inner_from = inner_scope.expression.args.get('from')
        if not inner_from:
            return False
        inner_from_table = inner_from.alias_or_name
        inner_projections = {s.alias_or_name: s for s in inner_scope.expression.selects}
        return any((col.table != inner_from_table for selection in selections for col in inner_projections[selection].find_all(exp.Column)))

    def _is_recursive():
        cte = inner_scope.expression.parent
        node = outer_scope.expression.parent
        while node:
            if node is cte:
                return True
            node = node.parent
        return False
    return isinstance(outer_scope.expression, exp.Select) and (not outer_scope.expression.is_star) and isinstance(inner_select, exp.Select) and (not any((inner_select.args.get(arg) for arg in UNMERGABLE_ARGS))) and inner_select.args.get('from') and (not outer_scope.pivots) and (not any((e.find(exp.AggFunc, exp.Select, exp.Explode) for e in inner_select.expressions))) and (not (leave_tables_isolated and len(outer_scope.selected_sources) > 1)) and (not (isinstance(from_or_join, exp.Join) and inner_select.args.get('where') and (from_or_join.side in ('FULL', 'LEFT', 'RIGHT')))) and (not (isinstance(from_or_join, exp.From) and inner_select.args.get('where') and any((j.side in ('FULL', 'RIGHT') for j in outer_scope.expression.args.get('joins', []))))) and (not _outer_select_joins_on_inner_select_join()) and (not _is_a_window_expression_in_unmergable_operation()) and (not _is_recursive()) and (not (inner_select.args.get('order') and outer_scope.is_union))