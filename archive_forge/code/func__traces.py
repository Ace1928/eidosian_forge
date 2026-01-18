from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
@_call_count('individuations')
def _traces(self, coloring: List[Color], stats: Optional[Stats]=None, depth: List[int]=[0]) -> List[Color]:
    if stats is not None and 'prunings' not in stats:
        stats['prunings'] = 0
    depth[0] += 1
    candidates = self._get_candidates(coloring)
    best: List[List[Color]] = []
    best_score = None
    best_experimental_score = None
    last_coloring = None
    generator: Dict[Node, Set[Node]] = defaultdict(set)
    visited: Set[Node] = set()
    for candidate, color in candidates:
        if candidate in generator:
            v = generator[candidate] & visited
            if len(v) > 0:
                visited.add(candidate)
                continue
        visited.add(candidate)
        coloring_copy: List[Color] = []
        color_copy = None
        for c in coloring:
            c_copy = c.copy()
            coloring_copy.append(c_copy)
            if c == color:
                color_copy = c_copy
        new_color = self._individuate(color_copy, candidate)
        coloring_copy.append(new_color)
        refined_coloring = self._refine(coloring_copy, [new_color])
        color_score = tuple([c.key() for c in refined_coloring])
        experimental = self._experimental_path(coloring_copy)
        experimental_score = set([c.key() for c in experimental])
        if last_coloring:
            generator = self._create_generator([last_coloring, experimental], generator)
        last_coloring = experimental
        if best_score is None or best_score < color_score:
            best = [refined_coloring]
            best_score = color_score
            best_experimental_score = experimental_score
        elif best_score > color_score:
            if stats is not None:
                stats['prunings'] += 1
        elif experimental_score != best_experimental_score:
            best.append(refined_coloring)
        elif stats is not None:
            stats['prunings'] += 1
    discrete: List[List[Color]] = [x for x in best if self._discrete(x)]
    if len(discrete) == 0:
        best_score = None
        best_depth = None
        for coloring in best:
            d = [depth[0]]
            new_color = self._traces(coloring, stats=stats, depth=d)
            color_score = tuple([c.key() for c in refined_coloring])
            if best_score is None or color_score > best_score:
                discrete = [new_color]
                best_score = color_score
                best_depth = d[0]
        depth[0] = best_depth
    return discrete[0]