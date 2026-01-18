from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
class DynamicChild(Resource):
    """
    A L{Resource} to be created on the fly by L{DynamicChildren}.
    """

    def __init__(self, path: bytes, request: IRequest) -> None:
        Resource.__init__(self)
        self.path = path
        self.request = request