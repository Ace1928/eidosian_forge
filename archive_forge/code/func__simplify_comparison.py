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
def _simplify_comparison(expression, left, right, or_=False):
    if isinstance(left, COMPARISONS) and isinstance(right, COMPARISONS):
        ll, lr = left.args.values()
        rl, rr = right.args.values()
        largs = {ll, lr}
        rargs = {rl, rr}
        matching = largs & rargs
        columns = {m for m in matching if not _is_constant(m) and (not m.find(*NONDETERMINISTIC))}
        if matching and columns:
            try:
                l = first(largs - columns)
                r = first(rargs - columns)
            except StopIteration:
                return expression
            if l.is_number and r.is_number:
                l = float(l.name)
                r = float(r.name)
            elif l.is_string and r.is_string:
                l = l.name
                r = r.name
            else:
                l = extract_date(l)
                if not l:
                    return None
                r = extract_date(r)
                if not r:
                    return None
                l, r = (cast_as_datetime(l), cast_as_datetime(r))
            for (a, av), (b, bv) in itertools.permutations(((left, l), (right, r))):
                if isinstance(a, LT_LTE) and isinstance(b, LT_LTE):
                    return left if (av > bv if or_ else av <= bv) else right
                if isinstance(a, GT_GTE) and isinstance(b, GT_GTE):
                    return left if (av < bv if or_ else av >= bv) else right
                if not or_:
                    if isinstance(a, exp.LT) and isinstance(b, GT_GTE):
                        if av <= bv:
                            return exp.false()
                    elif isinstance(a, exp.GT) and isinstance(b, LT_LTE):
                        if av >= bv:
                            return exp.false()
                    elif isinstance(a, exp.EQ):
                        if isinstance(b, exp.LT):
                            return exp.false() if av >= bv else a
                        if isinstance(b, exp.LTE):
                            return exp.false() if av > bv else a
                        if isinstance(b, exp.GT):
                            return exp.false() if av <= bv else a
                        if isinstance(b, exp.GTE):
                            return exp.false() if av < bv else a
                        if isinstance(b, exp.NEQ):
                            return exp.false() if av == bv else a
    return None