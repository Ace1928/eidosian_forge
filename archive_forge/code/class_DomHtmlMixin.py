from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
class DomHtmlMixin:
    CDATA_SECTION_NODE = Node.CDATA_SECTION_NODE
    ELEMENT_NODE = Node.ELEMENT_NODE
    TEXT_NODE = Node.TEXT_NODE
    _xp_childrennodes = XPath('child::node()')

    @property
    def documentElement(self):
        return self.getroottree().getroot()

    @property
    def nodeType(self):
        return Node.ELEMENT_NODE

    @property
    def nodeName(self):
        return self.tag

    @property
    def tagName(self):
        return self.tag

    @property
    def localName(self):
        return self.xpath('local-name(.)')

    def hasAttribute(self, name):
        return name in self.attrib

    def getAttribute(self, name):
        return self.get(name)

    def setAttribute(self, name, value):
        self.set(name, value)

    def cloneNode(self, deep):
        return deepcopy(self) if deep else copy(self)

    @property
    def attributes(self):
        attrs = {}
        for name, value in self.attrib.items():
            a = Attr(name)
            a.value = value
            attrs[name] = a
        return NamedNodeMap(attrs, {}, self)

    @property
    def parentNode(self):
        return self.getparent()

    @property
    def childNodes_xpath(self):
        for n in self._xp_childrennodes(self):
            if isinstance(n, ElementBase):
                yield n
            elif isinstance(n, (_ElementStringResult, _ElementUnicodeResult)):
                if isinstance(n, _ElementUnicodeResult):
                    n = DomElementUnicodeResult(n)
                else:
                    n.nodeType = Node.TEXT_NODE
                    n.data = n
                yield n

    @property
    def childNodes(self):
        if self.text:
            yield DomTextNode(self.text)
        for n in self.iterchildren():
            yield n
            if n.tail:
                yield DomTextNode(n.tail)

    def getElementsByTagName(self, name):
        return self.iterdescendants(name)

    def getElementById(self, i):
        return self.get_element_by_id(i)

    @property
    def data(self):
        if isinstance(self, (_ElementStringResult, _ElementUnicodeResult)):
            return self
        else:
            raise RuntimeError

    def toxml(self, encoding=None):
        return tostring(self, encoding=encoding if encoding is not None else 'unicode')