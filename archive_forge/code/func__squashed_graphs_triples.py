from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def _squashed_graphs_triples(g1: Graph, g2: Graph):
    for t1, t2 in zip(sorted(_squash_graph(g1)), sorted(_squash_graph(g2))):
        yield (t1, t2)