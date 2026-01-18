from __future__ import annotations
import collections
import functools
import operator
import typing
from functools import reduce
from typing import (
from pyparsing import ParseResults
from rdflib.paths import (
from rdflib.plugins.sparql.operators import TrueFilter, and_
from rdflib.plugins.sparql.operators import simplify as simplifyFilters
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import Prologue, Query, Update
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _traverseAgg(e, visitor: Callable[[Any, Any], Any]=lambda n, v: None):
    """
    Traverse a parse-tree, visit each node

    if visit functions return a value, replace current node
    """
    res = []
    if isinstance(e, (list, ParseResults, tuple)):
        res = [_traverseAgg(x, visitor) for x in e]
    elif isinstance(e, CompValue):
        for k, val in e.items():
            if val is not None:
                res.append(_traverseAgg(val, visitor))
    return visitor(e, res)