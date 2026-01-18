import io
import linecache
import warnings
from collections import OrderedDict
from html import escape
from typing import (
from xml.sax import handler, make_parser
from xml.sax.xmlreader import AttributesNSImpl, Locator
from zope.interface import implementer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import urlpath
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import fullyQualifiedName
from twisted.web import resource
from twisted.web._element import Element, renderer
from twisted.web._flatten import Flattenable, flatten, flattenString
from twisted.web._stan import CDATA, Comment, Tag, slot
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
def _getSourceLines(self):
    """
        Find the source line references by C{self.frame} and yield, in source
        line order, it and the previous and following lines.

        @return: A generator which yields two-tuples.  Each tuple gives a source
            line number and the contents of that source line.
        """
    filename = self.frame[1]
    lineNumber = self.frame[2]
    for snipLineNumber in range(lineNumber - 1, lineNumber + 2):
        yield (snipLineNumber, linecache.getline(filename, snipLineNumber).rstrip())