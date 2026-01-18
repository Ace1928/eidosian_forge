from __future__ import annotations
from typing import Any, Mapping, Optional, Union
from rdflib.graph import Graph
from rdflib.plugins.sparql.algebra import translateQuery, translateUpdate
from rdflib.plugins.sparql.evaluate import evalQuery
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
from rdflib.plugins.sparql.sparql import Query, Update
from rdflib.plugins.sparql.update import evalUpdate
from rdflib.query import Processor, Result, UpdateProcessor
from rdflib.term import Identifier
class SPARQLResult(Result):

    def __init__(self, res: Mapping[str, Any]):
        Result.__init__(self, res['type_'])
        self.vars = res.get('vars_')
        self.bindings = res.get('bindings')
        self.askAnswer = res.get('askAnswer')
        self.graph = res.get('graph')