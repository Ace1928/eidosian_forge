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
def _c(n):
    if isinstance(n, CompValue):
        if n.name in ('Builtin_EXISTS', 'Builtin_NOTEXISTS'):
            n.graph = translateGroupGraphPattern(n.graph)
            if n.graph.name == 'Filter':
                n.graph.no_isolated_scope = True