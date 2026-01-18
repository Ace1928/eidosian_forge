from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope
def _is_joined_on_all_unique_outputs(scope, join):
    unique_outputs = _unique_outputs(scope)
    if not unique_outputs:
        return False
    _, join_keys, _ = join_condition(join)
    remaining_unique_outputs = unique_outputs - {c.name for c in join_keys}
    return not remaining_unique_outputs