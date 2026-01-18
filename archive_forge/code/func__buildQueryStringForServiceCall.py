from __future__ import annotations
import collections
import itertools
import json as j
import re
from typing import (
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pyparsing import ParseException
from rdflib.graph import Graph
from rdflib.plugins.sparql import CUSTOM_EVALS, parser
from rdflib.plugins.sparql.aggregates import Aggregator
from rdflib.plugins.sparql.evalutils import (
from rdflib.plugins.sparql.parserutils import CompValue, value
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _buildQueryStringForServiceCall(ctx: QueryContext, service_query: str) -> str:
    try:
        parser.parseQuery(service_query)
    except ParseException:
        service_query = 'SELECT REDUCED * WHERE {' + service_query + '}'
        for p in ctx.prologue.namespace_manager.store.namespaces():
            service_query = 'PREFIX ' + p[0] + ':' + p[1].n3() + ' ' + service_query
        base = ctx.prologue.base
        if base is not None and len(base) > 0:
            service_query = 'BASE <' + base + '> ' + service_query
    sol = [v for v in ctx.solution() if isinstance(v, Variable)]
    if len(sol) > 0:
        variables = ' '.join([v.n3() for v in sol])
        variables_bound = ' '.join([ctx.get(v).n3() for v in sol])
        service_query = service_query + 'VALUES (' + variables + ') {(' + variables_bound + ')}'
    return service_query