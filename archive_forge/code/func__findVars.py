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
def _findVars(x, res: Set[Variable]) -> Optional[CompValue]:
    """
    Find all variables in a tree
    """
    if isinstance(x, Variable):
        res.add(x)
    if isinstance(x, CompValue):
        if x.name == 'Bind':
            res.add(x.var)
            return x
        elif x.name == 'SubSelect':
            if x.projection:
                res.update((v.var or v.evar for v in x.projection))
            return x