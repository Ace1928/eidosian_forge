from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope
def _should_eliminate_join(scope, join, alias):
    inner_source = scope.sources.get(alias)
    return isinstance(inner_source, Scope) and (not _join_is_used(scope, join, alias)) and (join.side == 'LEFT' and _is_joined_on_all_unique_outputs(inner_source, join) or (not join.args.get('on') and _has_single_output_row(inner_source)))