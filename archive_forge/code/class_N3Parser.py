from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
class N3Parser(TurtleParser):
    """
    An RDFLib parser for Notation3

    See http://www.w3.org/DesignIssues/Notation3.html

    """

    def __init__(self):
        pass

    def parse(self, source: InputSource, graph: Graph, encoding: Optional[str]='utf-8') -> None:
        ca = getattr(graph.store, 'context_aware', False)
        fa = getattr(graph.store, 'formula_aware', False)
        if not ca:
            raise ParserError('Cannot parse N3 into non-context-aware store.')
        elif not fa:
            raise ParserError('Cannot parse N3 into non-formula-aware store.')
        conj_graph = ConjunctiveGraph(store=graph.store)
        conj_graph.default_context = graph
        conj_graph.namespace_manager = graph.namespace_manager
        TurtleParser.parse(self, source, conj_graph, encoding, turtle=False)