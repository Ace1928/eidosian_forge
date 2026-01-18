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
class DummySink:

    def __init__(self):
        self.length = 0

    def triple(self, s, p, o):
        self.length += 1
        print(s, p, o)