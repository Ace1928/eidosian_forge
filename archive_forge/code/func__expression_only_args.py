from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _expression_only_args(expression: exp.Expression) -> t.List[exp.Expression]:
    args: t.List[t.Union[exp.Expression, t.List]] = []
    if expression:
        for a in expression.args.values():
            args.extend(ensure_list(a))
    return [a for a in args if isinstance(a, exp.Expression) and (not isinstance(a, IGNORED_LEAF_EXPRESSION_TYPES))]