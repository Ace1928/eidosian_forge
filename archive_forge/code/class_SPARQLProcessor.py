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
class SPARQLProcessor(Processor):

    def __init__(self, graph):
        self.graph = graph

    def query(self, strOrQuery: Union[str, Query], initBindings: Optional[Mapping[str, Identifier]]=None, initNs: Optional[Mapping[str, Any]]=None, base: Optional[str]=None, DEBUG: bool=False) -> Mapping[str, Any]:
        """
        Evaluate a query with the given initial bindings, and initial
        namespaces. The given base is used to resolve relative URIs in
        the query and will be overridden by any BASE given in the query.

        .. caution::

           This method can access indirectly requested network endpoints, for
           example, query processing will attempt to access network endpoints
           specified in ``SERVICE`` directives.

           When processing untrusted or potentially malicious queries, measures
           should be taken to restrict network and file access.

           For information on available security measures, see the RDFLib
           :doc:`Security Considerations </security_considerations>`
           documentation.
        """
        if isinstance(strOrQuery, str):
            strOrQuery = translateQuery(parseQuery(strOrQuery), base, initNs)
        return evalQuery(self.graph, strOrQuery, initBindings, base)