from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope

    Extract the join condition from a join expression.

    Args:
        join (exp.Join)
    Returns:
        tuple[list[str], list[str], exp.Expression]:
            Tuple of (source key, join key, remaining predicate)
    