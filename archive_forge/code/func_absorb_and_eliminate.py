from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def absorb_and_eliminate(expression, root=True):
    """
    absorption:
        A AND (A OR B) -> A
        A OR (A AND B) -> A
        A AND (NOT A OR B) -> A AND B
        A OR (NOT A AND B) -> A OR B
    elimination:
        (A AND B) OR (A AND NOT B) -> A
        (A OR B) AND (A OR NOT B) -> A
    """
    if isinstance(expression, exp.Connector) and (root or not expression.same_parent):
        kind = exp.Or if isinstance(expression, exp.And) else exp.And
        for a, b in itertools.permutations(expression.flatten(), 2):
            if isinstance(a, kind):
                aa, ab = a.unnest_operands()
                if is_complement(b, aa):
                    aa.replace(exp.true() if kind == exp.And else exp.false())
                elif is_complement(b, ab):
                    ab.replace(exp.true() if kind == exp.And else exp.false())
                elif (set(b.flatten()) if isinstance(b, kind) else {b}) < set(a.flatten()):
                    a.replace(exp.false() if kind == exp.And else exp.true())
                elif isinstance(b, kind):
                    rhs = b.unnest_operands()
                    ba, bb = rhs
                    if aa in rhs and (is_complement(ab, ba) or is_complement(ab, bb)):
                        a.replace(aa)
                        b.replace(aa)
                    elif ab in rhs and (is_complement(aa, ba) or is_complement(aa, bb)):
                        a.replace(ab)
                        b.replace(ab)
    return expression