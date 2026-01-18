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
def _flatsaxParse(fl: Union[IO[AnyStr], str]) -> List['Flattenable']:
    """
    Perform a SAX parse of an XML document with the _ToStan class.

    @param fl: The XML document to be parsed.

    @return: a C{list} of Stan objects.
    """
    parser = make_parser()
    parser.setFeature(handler.feature_validation, 0)
    parser.setFeature(handler.feature_namespaces, 1)
    parser.setFeature(handler.feature_external_ges, 0)
    parser.setFeature(handler.feature_external_pes, 0)
    s = _ToStan(getattr(fl, 'name', None))
    parser.setContentHandler(s)
    parser.setEntityResolver(s)
    parser.setProperty(handler.property_lexical_handler, s)
    parser.parse(fl)
    return s.document