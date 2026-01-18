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
@implementer(ITemplateLoader)
class XMLString:
    """
    An L{ITemplateLoader} that loads and parses XML from a string.
    """

    def __init__(self, s: Union[str, bytes]):
        """
        Run the parser on a L{io.StringIO} copy of the string.

        @param s: The string from which to load the XML.
        @type s: L{str}, or a UTF-8 encoded L{bytes}.
        """
        if not isinstance(s, str):
            s = s.decode('utf8')
        self._loadedTemplate: List['Flattenable'] = _flatsaxParse(io.StringIO(s))
        'The loaded document.'

    def load(self) -> List['Flattenable']:
        """
        Return the document.

        @return: the loaded document.
        """
        return self._loadedTemplate