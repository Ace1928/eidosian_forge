from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
class ImplicitAllowedMethods(Resource):
    """
    A L{Resource} which implicitly defines its allowed methods by defining
    renderers to handle them.
    """

    def render_GET(self, request: object) -> None:
        pass

    def render_PUT(self, request: object) -> None:
        pass