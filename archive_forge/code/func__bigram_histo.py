from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _bigram_histo(self, expression: exp.Expression) -> t.DefaultDict[str, int]:
    if id(expression) in self._bigram_histo_cache:
        return self._bigram_histo_cache[id(expression)]
    expression_str = self._sql_generator.generate(expression)
    count = max(0, len(expression_str) - 1)
    bigram_histo: t.DefaultDict[str, int] = defaultdict(int)
    for i in range(count):
        bigram_histo[expression_str[i:i + 2]] += 1
    self._bigram_histo_cache[id(expression)] = bigram_histo
    return bigram_histo