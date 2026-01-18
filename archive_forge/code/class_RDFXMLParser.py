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
class RDFXMLParser(Parser):

    def __init__(self):
        pass

    def parse(self, source: InputSource, sink: Graph, **args: Any) -> None:
        self._parser = create_parser(source, sink)
        content_handler = self._parser.getContentHandler()
        preserve_bnode_ids = args.get('preserve_bnode_ids', None)
        if preserve_bnode_ids is not None:
            content_handler.preserve_bnode_ids = preserve_bnode_ids
        self._parser.parse(source)