from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _parent_similarity_score(source: t.Optional[exp.Expression], target: t.Optional[exp.Expression]) -> int:
    if source is None or target is None or type(source) is not type(target):
        return 0
    return 1 + _parent_similarity_score(source.parent, target.parent)