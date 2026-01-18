from __future__ import annotations
import collections
import datetime
import itertools
import typing as t
from collections.abc import Mapping, MutableMapping
from typing import (
import isodate
import rdflib.plugins.sparql
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import NamespaceManager
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import BNode, Identifier, Literal, Node, URIRef, Variable
class AlreadyBound(SPARQLError):
    """Raised when trying to bind a variable that is already bound!"""

    def __init__(self):
        SPARQLError.__init__(self)