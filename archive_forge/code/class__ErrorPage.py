from typing import cast
from twisted.web import http
from twisted.web.iweb import IRenderable, IRequest
from twisted.web.resource import IResource, Resource
from twisted.web.template import renderElement, tags
class _ErrorPage(Resource):
    """
    L{_ErrorPage} is a resource that responds to all requests with a particular
    (parameterized) HTTP status code and an HTML body containing some
    descriptive text. This is useful for rendering simple error pages.

    @see: L{twisted.web.pages.errorPage}

    @ivar _code: An integer HTTP status code which will be used for the
        response.

    @ivar _brief: A short string which will be included in the response body as
        the page title.

    @ivar _detail: A longer string which will be included in the response body.
    """

    def __init__(self, code: int, brief: str, detail: str) -> None:
        super().__init__()
        self._code: int = code
        self._brief: str = brief
        self._detail: str = detail

    def render(self, request: IRequest) -> object:
        """
        Respond to all requests with the given HTTP status code and an HTML
        document containing the explanatory strings.
        """
        request.setResponseCode(self._code)
        request.setHeader(b'content-type', b'text/html; charset=utf-8')
        return renderElement(request, cast(IRenderable, tags.html(tags.head(tags.title(f'{self._code} - {self._brief}')), tags.body(tags.h1(self._brief), tags.p(self._detail)))))

    def getChild(self, path: bytes, request: IRequest) -> Resource:
        """
        Handle all requests for which L{_ErrorPage} lacks a child by returning
        this error page.

        @param path: A path segment.

        @param request: HTTP request
        """
        return self