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
def _find_first_child_projections(M: CompValue) -> Iterable[CompValue]:
    """
    Recursively find the first child instance of a Projection operation in each of
      the branches of the query execution plan/tree.
    """
    for child_op in M.values():
        if isinstance(child_op, CompValue):
            if child_op.name == 'Project':
                yield child_op
            else:
                for child_projection in _find_first_child_projections(child_op):
                    yield child_projection