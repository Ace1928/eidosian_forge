from __future__ import annotations
import logging
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import while_changing
from sqlglot.optimizer.scope import find_all_in_scope
from sqlglot.optimizer.simplify import flatten, rewrite_between, uniq_sort
def _predicate_lengths(expression, dnf):
    """
    Returns a list of predicate lengths when expanded to normalized form.

    (A AND B) OR C -> [2, 2] because len(A OR C), len(B OR C).
    """
    expression = expression.unnest()
    if not isinstance(expression, exp.Connector):
        return (1,)
    left, right = expression.args.values()
    if isinstance(expression, exp.And if dnf else exp.Or):
        return tuple((a + b for a in _predicate_lengths(left, dnf) for b in _predicate_lengths(right, dnf)))
    return _predicate_lengths(left, dnf) + _predicate_lengths(right, dnf)