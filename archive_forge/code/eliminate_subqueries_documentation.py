import itertools
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import build_scope

    Returns:
        tuple of (name, cte)
        where `name` is a new name for this CTE in the root scope and `cte` is a new CTE instance.
        If this CTE duplicates an existing CTE, `cte` will be None.
    