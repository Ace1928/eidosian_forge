from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
class DomHtmlElementClassLookup(HtmlElementClassLookup):

    def __init__(self):
        super().__init__()
        self._lookups = {}

    def lookup(self, node_type, document, namespace, name):
        k = (node_type, document, namespace, name)
        t = self._lookups.get(k)
        if t is None:
            cur = super().lookup(node_type, document, namespace, name)
            newtype = type('Dom' + cur.__name__, (cur, DomHtmlMixin), {})
            self._lookups[k] = newtype
            return newtype
        else:
            return t