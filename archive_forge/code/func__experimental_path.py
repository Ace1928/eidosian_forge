from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def _experimental_path(self, coloring: List[Color]) -> List[Color]:
    coloring = [c.copy() for c in coloring]
    while not self._discrete(coloring):
        color = [x for x in coloring if not x.discrete()][0]
        node = color.nodes[0]
        new_color = self._individuate(color, node)
        coloring.append(new_color)
        coloring = self._refine(coloring, [new_color])
    return coloring