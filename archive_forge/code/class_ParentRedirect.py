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
class ParentRedirect(resource.Resource):
    """
    Redirect to the nearest directory and strip any query string.

    This generates redirects like::

        /              →  /
        /foo           →  /
        /foo?bar       →  /
        /foo/          →  /foo/
        /foo/bar       →  /foo/
        /foo/bar?baz   →  /foo/

    However, the generated I{Location} header contains an absolute URL rather
    than a path.

    The response is the same regardless of HTTP method.
    """
    isLeaf = 1

    def render(self, request: IRequest) -> bytes:
        """
        Respond to all requests by redirecting to nearest directory.
        """
        here = str(urlpath.URLPath.fromRequest(request).here()).encode('ascii')
        return redirectTo(here, request)