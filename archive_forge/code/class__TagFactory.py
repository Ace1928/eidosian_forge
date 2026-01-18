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
class _TagFactory:
    """
    A factory for L{Tag} objects; the implementation of the L{tags} object.

    This allows for the syntactic convenience of C{from twisted.web.template
    import tags; tags.a(href="linked-page.html")}, where 'a' can be basically
    any HTML tag.

    The class is not exposed publicly because you only ever need one of these,
    and we already made it for you.

    @see: L{tags}
    """

    def __getattr__(self, tagName: str) -> Tag:
        if tagName == 'transparent':
            return Tag('')
        tagName = tagName.rstrip('_')
        if tagName not in VALID_HTML_TAG_NAMES:
            raise AttributeError(f'unknown tag {tagName!r}')
        return Tag(tagName)