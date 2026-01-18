from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def _canonicalize_bnodes(self, triple: '_TripleType', labels: Dict[Node, str]):
    for term in triple:
        if isinstance(term, BNode):
            yield BNode(value='cb%s' % labels[term])
        else:
            yield term