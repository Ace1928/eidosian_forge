from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _merge_from(outer_scope, inner_scope, node_to_replace, alias):
    """
    Merge FROM clause of inner query into outer query.

    Args:
        outer_scope (sqlglot.optimizer.scope.Scope)
        inner_scope (sqlglot.optimizer.scope.Scope)
        node_to_replace (exp.Subquery|exp.Table)
        alias (str)
    """
    new_subquery = inner_scope.expression.args['from'].this
    new_subquery.set('joins', node_to_replace.args.get('joins'))
    node_to_replace.replace(new_subquery)
    for join_hint in outer_scope.join_hints:
        tables = join_hint.find_all(exp.Table)
        for table in tables:
            if table.alias_or_name == node_to_replace.alias_or_name:
                table.set('this', exp.to_identifier(new_subquery.alias_or_name))
    outer_scope.remove_source(alias)
    outer_scope.add_source(new_subquery.alias_or_name, inner_scope.sources[new_subquery.alias_or_name])