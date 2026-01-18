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
class TagLoaderTests(FlattenTestCase):
    """
    Tests for L{TagLoader}.
    """

    def setUp(self) -> None:
        self.loader = TagLoader(tags.i('test'))

    def test_interface(self) -> None:
        """
        An instance of L{TagLoader} provides L{ITemplateLoader}.
        """
        self.assertTrue(verifyObject(ITemplateLoader, self.loader))

    def test_loadsList(self) -> None:
        """
        L{TagLoader.load} returns a list, per L{ITemplateLoader}.
        """
        self.assertIsInstance(self.loader.load(), list)

    def test_flatten(self) -> None:
        """
        L{TagLoader} can be used in an L{Element}, and flattens as the tag used
        to construct the L{TagLoader} would flatten.
        """
        e = Element(self.loader)
        self.assertFlattensImmediately(e, b'<i>test</i>')