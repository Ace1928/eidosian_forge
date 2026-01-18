from __future__ import annotations
import warnings
from typing import Sequence
from zope.interface import Attribute, Interface, implementer
from incremental import Version
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import prefixedMethodNames
from twisted.web._responses import FORBIDDEN, NOT_FOUND
from twisted.web.error import UnsupportedMethod
class _UnsafeErrorPage(Resource):
    """
    L{_UnsafeErrorPage}, publicly available via the deprecated alias
    C{ErrorPage}, is a resource which responds with a particular
    (parameterized) status and a body consisting of HTML containing some
    descriptive text.  This is useful for rendering simple error pages.

    Deprecated in Twisted 22.10.0 because it permits HTML injection; use
    L{twisted.web.pages.errorPage} instead.

    @ivar template: A native string which will have a dictionary interpolated
        into it to generate the response body.  The dictionary has the following
        keys:

          - C{"code"}: The status code passed to L{_UnsafeErrorPage.__init__}.
          - C{"brief"}: The brief description passed to
            L{_UnsafeErrorPage.__init__}.
          - C{"detail"}: The detailed description passed to
            L{_UnsafeErrorPage.__init__}.

    @ivar code: An integer status code which will be used for the response.
    @type code: C{int}

    @ivar brief: A short string which will be included in the response body as
        the page title.
    @type brief: C{str}

    @ivar detail: A longer string which will be included in the response body.
    @type detail: C{str}
    """
    template = '\n<html>\n  <head><title>%(code)s - %(brief)s</title></head>\n  <body>\n    <h1>%(brief)s</h1>\n    <p>%(detail)s</p>\n  </body>\n</html>\n'

    def __init__(self, status, brief, detail):
        Resource.__init__(self)
        self.code = status
        self.brief = brief
        self.detail = detail

    def render(self, request):
        request.setResponseCode(self.code)
        request.setHeader(b'content-type', b'text/html; charset=utf-8')
        interpolated = self.template % dict(code=self.code, brief=self.brief, detail=self.detail)
        if isinstance(interpolated, str):
            return interpolated.encode('utf-8')
        return interpolated

    def getChild(self, chnam, request):
        return self