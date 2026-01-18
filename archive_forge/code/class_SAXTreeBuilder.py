from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
class SAXTreeBuilder(TreeBuilder):
    """A Beautiful Soup treebuilder that listens for SAX events.

    This is not currently used for anything, but it demonstrates
    how a simple TreeBuilder would work.
    """

    def feed(self, markup):
        raise NotImplementedError()

    def close(self):
        pass

    def startElement(self, name, attrs):
        attrs = dict(((key[1], value) for key, value in list(attrs.items())))
        self.soup.handle_starttag(name, attrs)

    def endElement(self, name):
        self.soup.handle_endtag(name)

    def startElementNS(self, nsTuple, nodeName, attrs):
        self.startElement(nodeName, attrs)

    def endElementNS(self, nsTuple, nodeName):
        self.endElement(nodeName)

    def startPrefixMapping(self, prefix, nodeValue):
        pass

    def endPrefixMapping(self, prefix):
        pass

    def characters(self, content):
        self.soup.handle_data(content)

    def startDocument(self):
        pass

    def endDocument(self):
        pass