from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
class _TripleCanonicalizer:

    def __init__(self, graph: Graph, hashfunc: _HashT=sha256):
        self.graph = graph

        def _hashfunc(s: str):
            h = hashfunc()
            h.update(str(s).encode('utf8'))
            return int(h.hexdigest(), 16)
        self._hash_cache: HashCache = {}
        self.hashfunc = _hashfunc

    def _discrete(self, coloring: List[Color]) -> bool:
        return len([c for c in coloring if not c.discrete()]) == 0

    def _initial_color(self) -> List[Color]:
        """Finds an initial color for the graph.

        Finds an initial color of the graph by finding all blank nodes and
        non-blank nodes that are adjacent. Nodes that are not adjacent to blank
        nodes are not included, as they are a) already colored (by URI or literal)
        and b) do not factor into the color of any blank node.
        """
        bnodes: Set[BNode] = set()
        others = set()
        self._neighbors = defaultdict(set)
        for s, p, o in self.graph:
            nodes = set([s, p, o])
            b = set([x for x in nodes if isinstance(x, BNode)])
            if len(b) > 0:
                others |= nodes - b
                bnodes |= b
                if isinstance(s, BNode):
                    self._neighbors[s].add(o)
                if isinstance(o, BNode):
                    self._neighbors[o].add(s)
                if isinstance(p, BNode):
                    self._neighbors[p].add(s)
                    self._neighbors[p].add(p)
        if len(bnodes) > 0:
            return [Color(list(bnodes), self.hashfunc, hash_cache=self._hash_cache)] + [Color([x], self.hashfunc, x, hash_cache=self._hash_cache) for x in others]
        else:
            return []

    def _individuate(self, color, individual):
        new_color = list(color.color)
        new_color.append((len(color.nodes),))
        color.nodes.remove(individual)
        c = Color([individual], self.hashfunc, tuple(new_color), hash_cache=self._hash_cache)
        return c

    def _get_candidates(self, coloring: List[Color]) -> Iterator[Tuple[Node, Color]]:
        for c in [c for c in coloring if not c.discrete()]:
            for node in c.nodes:
                yield (node, c)

    def _refine(self, coloring: List[Color], sequence: List[Color]) -> List[Color]:
        sequence = sorted(sequence, key=lambda x: x.key(), reverse=True)
        coloring = coloring[:]
        while len(sequence) > 0 and (not self._discrete(coloring)):
            W = sequence.pop()
            for c in coloring[:]:
                if len(c.nodes) > 1 or isinstance(c.nodes[0], BNode):
                    colors = sorted(c.distinguish(W, self.graph), key=lambda x: x.key(), reverse=True)
                    coloring.remove(c)
                    coloring.extend(colors)
                    try:
                        si = sequence.index(c)
                        sequence = sequence[:si] + colors + sequence[si + 1:]
                    except ValueError:
                        sequence = colors[1:] + sequence
        combined_colors: List[Color] = []
        combined_color_map: Dict[str, Color] = dict()
        for color in coloring:
            color_hash = color.hash_color()
            if color_hash in combined_color_map:
                combined_color_map[color_hash].nodes.extend(color.nodes)
            else:
                combined_colors.append(color)
                combined_color_map[color_hash] = color
        return combined_colors

    @_runtime('to_hash_runtime')
    def to_hash(self, stats: Optional[Stats]=None):
        result = 0
        for triple in self.canonical_triples(stats=stats):
            result += self.hashfunc(' '.join([x.n3() for x in triple]))
        if stats is not None:
            stats['graph_digest'] = '%x' % result
        return result

    def _experimental_path(self, coloring: List[Color]) -> List[Color]:
        coloring = [c.copy() for c in coloring]
        while not self._discrete(coloring):
            color = [x for x in coloring if not x.discrete()][0]
            node = color.nodes[0]
            new_color = self._individuate(color, node)
            coloring.append(new_color)
            coloring = self._refine(coloring, [new_color])
        return coloring

    def _create_generator(self, colorings: List[List[Color]], groupings: Optional[Dict[Node, Set[Node]]]=None) -> Dict[Node, Set[Node]]:
        if not groupings:
            groupings = defaultdict(set)
        for group in zip(*colorings):
            g = set([c.nodes[0] for c in group])
            for n in group:
                g |= groupings[n]
            for n in g:
                groupings[n] = g
        return groupings

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

    def canonical_triples(self, stats: Optional[Stats]=None):
        if stats is not None:
            start_coloring = datetime.now()
        coloring = self._initial_color()
        if stats is not None:
            stats['triple_count'] = len(self.graph)
            stats['adjacent_nodes'] = max(0, len(coloring) - 1)
        coloring = self._refine(coloring, coloring[:])
        if stats is not None:
            stats['initial_coloring_runtime'] = _total_seconds(datetime.now() - start_coloring)
            stats['initial_color_count'] = len(coloring)
        if not self._discrete(coloring):
            depth = [0]
            coloring = self._traces(coloring, stats=stats, depth=depth)
            if stats is not None:
                stats['tree_depth'] = depth[0]
        elif stats is not None:
            stats['individuations'] = 0
            stats['tree_depth'] = 0
        if stats is not None:
            stats['color_count'] = len(coloring)
        bnode_labels: Dict[Node, str] = dict([(c.nodes[0], c.hash_color()) for c in coloring])
        if stats is not None:
            stats['canonicalize_triples_runtime'] = _total_seconds(datetime.now() - start_coloring)
        for triple in self.graph:
            result = tuple(self._canonicalize_bnodes(triple, bnode_labels))
            yield result

    def _canonicalize_bnodes(self, triple: '_TripleType', labels: Dict[Node, str]):
        for term in triple:
            if isinstance(term, BNode):
                yield BNode(value='cb%s' % labels[term])
            else:
                yield term