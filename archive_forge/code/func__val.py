from __future__ import annotations
import collections
from typing import (
from rdflib.plugins.sparql.operators import EBV
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _val(v: _ValueT) -> Tuple[int, _ValueT]:
    """utilitity for ordering things"""
    if isinstance(v, Variable):
        return (0, v)
    elif isinstance(v, BNode):
        return (1, v)
    elif isinstance(v, URIRef):
        return (2, v)
    elif isinstance(v, Literal):
        return (3, v)