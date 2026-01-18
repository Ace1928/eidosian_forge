from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _dice_coefficient(self, source: exp.Expression, target: exp.Expression) -> float:
    source_histo = self._bigram_histo(source)
    target_histo = self._bigram_histo(target)
    total_grams = sum(source_histo.values()) + sum(target_histo.values())
    if not total_grams:
        return 1.0 if source == target else 0.0
    overlap_len = 0
    overlapping_grams = set(source_histo) & set(target_histo)
    for g in overlapping_grams:
        overlap_len += min(source_histo[g], target_histo[g])
    return 2 * overlap_len / total_grams