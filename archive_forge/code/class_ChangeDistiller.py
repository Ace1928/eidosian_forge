from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
class ChangeDistiller:
    """
    The implementation of the Change Distiller algorithm described by Beat Fluri and Martin Pinzger in
    their paper https://ieeexplore.ieee.org/document/4339230, which in turn is based on the algorithm by
    Chawathe et al. described in http://ilpubs.stanford.edu:8090/115/1/1995-46.pdf.
    """

    def __init__(self, f: float=0.6, t: float=0.6) -> None:
        self.f = f
        self.t = t
        self._sql_generator = Dialect().generator()

    def diff(self, source: exp.Expression, target: exp.Expression, matchings: t.List[t.Tuple[exp.Expression, exp.Expression]] | None=None) -> t.List[Edit]:
        matchings = matchings or []
        pre_matched_nodes = {id(s): id(t) for s, t in matchings}
        if len({n for pair in pre_matched_nodes.items() for n in pair}) != 2 * len(matchings):
            raise ValueError('Each node can be referenced at most once in the list of matchings')
        self._source = source
        self._target = target
        self._source_index = {id(n): n for n in self._source.bfs() if not isinstance(n, IGNORED_LEAF_EXPRESSION_TYPES)}
        self._target_index = {id(n): n for n in self._target.bfs() if not isinstance(n, IGNORED_LEAF_EXPRESSION_TYPES)}
        self._unmatched_source_nodes = set(self._source_index) - set(pre_matched_nodes)
        self._unmatched_target_nodes = set(self._target_index) - set(pre_matched_nodes.values())
        self._bigram_histo_cache: t.Dict[int, t.DefaultDict[str, int]] = {}
        matching_set = self._compute_matching_set() | {(s, t) for s, t in pre_matched_nodes.items()}
        return self._generate_edit_script(matching_set)

    def _generate_edit_script(self, matching_set: t.Set[t.Tuple[int, int]]) -> t.List[Edit]:
        edit_script: t.List[Edit] = []
        for removed_node_id in self._unmatched_source_nodes:
            edit_script.append(Remove(self._source_index[removed_node_id]))
        for inserted_node_id in self._unmatched_target_nodes:
            edit_script.append(Insert(self._target_index[inserted_node_id]))
        for kept_source_node_id, kept_target_node_id in matching_set:
            source_node = self._source_index[kept_source_node_id]
            target_node = self._target_index[kept_target_node_id]
            if not isinstance(source_node, UPDATABLE_EXPRESSION_TYPES) or source_node == target_node:
                edit_script.extend(self._generate_move_edits(source_node, target_node, matching_set))
                edit_script.append(Keep(source_node, target_node))
            else:
                edit_script.append(Update(source_node, target_node))
        return edit_script

    def _generate_move_edits(self, source: exp.Expression, target: exp.Expression, matching_set: t.Set[t.Tuple[int, int]]) -> t.List[Move]:
        source_args = [id(e) for e in _expression_only_args(source)]
        target_args = [id(e) for e in _expression_only_args(target)]
        args_lcs = set(_lcs(source_args, target_args, lambda l, r: (l, r) in matching_set))
        move_edits = []
        for a in source_args:
            if a not in args_lcs and a not in self._unmatched_source_nodes:
                move_edits.append(Move(self._source_index[a]))
        return move_edits

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

    def _compute_leaf_matching_set(self) -> t.Set[t.Tuple[int, int]]:
        candidate_matchings: t.List[t.Tuple[float, int, int, exp.Expression, exp.Expression]] = []
        source_leaves = list(_get_leaves(self._source))
        target_leaves = list(_get_leaves(self._target))
        for source_leaf in source_leaves:
            for target_leaf in target_leaves:
                if _is_same_type(source_leaf, target_leaf):
                    similarity_score = self._dice_coefficient(source_leaf, target_leaf)
                    if similarity_score >= self.f:
                        heappush(candidate_matchings, (-similarity_score, -_parent_similarity_score(source_leaf, target_leaf), len(candidate_matchings), source_leaf, target_leaf))
        matching_set = set()
        while candidate_matchings:
            _, _, _, source_leaf, target_leaf = heappop(candidate_matchings)
            if id(source_leaf) in self._unmatched_source_nodes and id(target_leaf) in self._unmatched_target_nodes:
                matching_set.add((id(source_leaf), id(target_leaf)))
                self._unmatched_source_nodes.remove(id(source_leaf))
                self._unmatched_target_nodes.remove(id(target_leaf))
        return matching_set

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