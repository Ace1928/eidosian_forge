import sys
from io import StringIO
from typing import List, Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, succeed
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.trial.util import suppress as SUPPRESS
from twisted.web._element import UnexposedMethodError
from twisted.web.error import FlattenerError, MissingRenderMethod, MissingTemplateLoader
from twisted.web.iweb import IRequest, ITemplateLoader
from twisted.web.server import NOT_DONE_YET
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
from twisted.web.test.test_web import DummyRequest
@implementer(IRequest)
class DummyRenderRequest(DummyRequest):
    """
    A dummy request object that has a C{site} attribute.

    This does not implement the full IRequest interface, but enough of it
    for this test suite.
    """

    def __init__(self) -> None:
        super().__init__([b''])
        self.site = FakeSite()