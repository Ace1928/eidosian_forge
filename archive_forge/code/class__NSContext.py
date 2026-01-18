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
class _NSContext:
    """
    A mapping from XML namespaces onto their prefixes in the document.
    """

    def __init__(self, parent: Optional['_NSContext']=None):
        """
        Pull out the parent's namespaces, if there's no parent then default to
        XML.
        """
        self.parent = parent
        if parent is not None:
            self.nss: Dict[Optional[str], Optional[str]] = OrderedDict(parent.nss)
        else:
            self.nss = {'http://www.w3.org/XML/1998/namespace': 'xml'}

    def get(self, k: Optional[str], d: Optional[str]=None) -> Optional[str]:
        """
        Get a prefix for a namespace.

        @param d: The default prefix value.
        """
        return self.nss.get(k, d)

    def __setitem__(self, k: Optional[str], v: Optional[str]) -> None:
        """
        Proxy through to setting the prefix for the namespace.
        """
        self.nss.__setitem__(k, v)

    def __getitem__(self, k: Optional[str]) -> Optional[str]:
        """
        Proxy through to getting the prefix for the namespace.
        """
        return self.nss.__getitem__(k)