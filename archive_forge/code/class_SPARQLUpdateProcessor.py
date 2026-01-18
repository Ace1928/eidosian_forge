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
class SPARQLUpdateProcessor(UpdateProcessor):

    def __init__(self, graph):
        self.graph = graph

    def update(self, strOrQuery: Union[str, Update], initBindings: Optional[Mapping[str, Identifier]]=None, initNs: Optional[Mapping[str, Any]]=None) -> None:
        """
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
            strOrQuery = translateUpdate(parseUpdate(strOrQuery), initNs=initNs)
        return evalUpdate(self.graph, strOrQuery, initBindings)