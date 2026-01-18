from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
class _call_count:

    def __init__(self, label):
        self.label = label

    def __call__(self, f):
        if self.label is None:
            self.label = f.__name__ + '_runtime'

        def wrapped_f(*args, **kwargs):
            if 'stats' in kwargs and kwargs['stats'] is not None:
                stats = kwargs['stats']
                if self.label not in stats:
                    stats[self.label] = 0
                stats[self.label] += 1
            return f(*args, **kwargs)
        return wrapped_f