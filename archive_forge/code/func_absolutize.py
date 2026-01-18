from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple
from urllib.parse import urldefrag, urljoin
from xml.sax import handler, make_parser, xmlreader
from xml.sax.handler import ErrorHandler
from xml.sax.saxutils import escape, quoteattr
from rdflib.exceptions import Error, ParserError
from rdflib.graph import Graph
from rdflib.namespace import RDF, is_ncname
from rdflib.parser import InputSource, Parser
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.term import BNode, Identifier, Literal, URIRef
def absolutize(self, uri: str) -> URIRef:
    result = urljoin(self.current.base, uri, allow_fragments=1)
    if uri and uri[-1] == '#' and (result[-1] != '#'):
        result = '%s#' % result
    return URIRef(result)