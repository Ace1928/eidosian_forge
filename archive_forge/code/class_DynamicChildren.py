from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
class DynamicChildren(Resource):
    """
    A L{Resource} with dynamic children.
    """

    def getChild(self, path: bytes, request: IRequest) -> DynamicChild:
        return DynamicChild(path, request)