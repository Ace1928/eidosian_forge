from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify
def _replace_alias(column):
    if isinstance(column, exp.Column) and column.name in aliases:
        return aliases[column.name].copy()
    return column