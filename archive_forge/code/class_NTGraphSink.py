from __future__ import annotations
import codecs
import re
from io import BytesIO, StringIO, TextIOBase
from typing import (
from rdflib.compat import _string_escape_map, decodeUnicodeEscape
from rdflib.exceptions import ParserError as ParseError
from rdflib.parser import InputSource, Parser
from rdflib.term import BNode as bNode
from rdflib.term import Literal
from rdflib.term import URIRef
from rdflib.term import URIRef as URI
class NTGraphSink:
    __slots__ = ('g',)

    def __init__(self, graph: 'Graph'):
        self.g = graph

    def triple(self, s: '_SubjectType', p: '_PredicateType', o: '_ObjectType') -> None:
        self.g.add((s, p, o))