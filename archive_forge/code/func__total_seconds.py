from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def _total_seconds(td):
    result = td.days * 24 * 60 * 60
    result += td.seconds
    result += td.microseconds / 1000000.0
    return result