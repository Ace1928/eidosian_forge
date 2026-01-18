from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
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