from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence
from rdflib.graph import Graph
from rdflib.plugins.sparql.evaluate import evalBGP, evalPart
from rdflib.plugins.sparql.evalutils import _fillTemplate, _join
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenDict, QueryContext, Update
from rdflib.term import Identifier, URIRef, Variable
def _graphOrDefault(ctx: QueryContext, g: str) -> Optional[Graph]:
    if g == 'DEFAULT':
        return ctx.graph
    else:
        return ctx.dataset.get_context(g)