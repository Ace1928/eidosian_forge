from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
class DomElementUnicodeResult:
    CDATA_SECTION_NODE = Node.CDATA_SECTION_NODE
    ELEMENT_NODE = Node.ELEMENT_NODE
    TEXT_NODE = Node.TEXT_NODE

    def __init__(self, text):
        self.text = text
        self.nodeType = Node.TEXT_NODE

    @property
    def data(self):
        if isinstance(self.text, _ElementUnicodeResult):
            return self.text
        else:
            raise RuntimeError