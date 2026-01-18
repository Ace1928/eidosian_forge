from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def _compute_matching_set(self) -> t.Set[t.Tuple[int, int]]:
    leaves_matching_set = self._compute_leaf_matching_set()
    matching_set = leaves_matching_set.copy()
    ordered_unmatched_source_nodes = {id(n): None for n in self._source.bfs() if id(n) in self._unmatched_source_nodes}
    ordered_unmatched_target_nodes = {id(n): None for n in self._target.bfs() if id(n) in self._unmatched_target_nodes}
    for source_node_id in ordered_unmatched_source_nodes:
        for target_node_id in ordered_unmatched_target_nodes:
            source_node = self._source_index[source_node_id]
            target_node = self._target_index[target_node_id]
            if _is_same_type(source_node, target_node):
                source_leaf_ids = {id(l) for l in _get_leaves(source_node)}
                target_leaf_ids = {id(l) for l in _get_leaves(target_node)}
                max_leaves_num = max(len(source_leaf_ids), len(target_leaf_ids))
                if max_leaves_num:
                    common_leaves_num = sum((1 if s in source_leaf_ids and t in target_leaf_ids else 0 for s, t in leaves_matching_set))
                    leaf_similarity_score = common_leaves_num / max_leaves_num
                else:
                    leaf_similarity_score = 0.0
                adjusted_t = self.t if min(len(source_leaf_ids), len(target_leaf_ids)) > 4 else 0.4
                if leaf_similarity_score >= 0.8 or (leaf_similarity_score >= adjusted_t and self._dice_coefficient(source_node, target_node) >= self.f):
                    matching_set.add((source_node_id, target_node_id))
                    self._unmatched_source_nodes.remove(source_node_id)
                    self._unmatched_target_nodes.remove(target_node_id)
                    ordered_unmatched_target_nodes.pop(target_node_id, None)
                    break
    return matching_set