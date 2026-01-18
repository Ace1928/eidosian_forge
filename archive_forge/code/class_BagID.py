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
class BagID(URIRef):
    __slots__ = ['li']

    def __init__(self, val):
        super(URIRef, self).__init__(val)
        self.li = 0

    def next_li(self):
        self.li += 1
        return RDFNS['_%s' % self.li]