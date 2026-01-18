from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
class XmlDomHTMLParser(HTMLParser):
    """An HTML parser that is configured to return XmlDomHtmlElement
    objects, compatible with xml.dom API
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        parser_lookup = DomHtmlElementClassLookup()
        self.set_element_class_lookup(parser_lookup)