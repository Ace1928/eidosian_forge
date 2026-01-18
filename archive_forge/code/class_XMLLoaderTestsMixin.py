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
class XMLLoaderTestsMixin:
    deprecatedUse: bool
    '\n    C{True} if this use of L{XMLFile} is deprecated and should emit\n    a C{DeprecationWarning}.\n    '
    templateString = '<p>Hello, world.</p>'
    '\n    Simple template to use to exercise the loaders.\n    '

    def loaderFactory(self) -> ITemplateLoader:
        raise NotImplementedError

    def test_load(self) -> None:
        """
        Verify that the loader returns a tag with the correct children.
        """
        assert isinstance(self, TestCase)
        loader = self.loaderFactory()
        tag, = loader.load()
        assert isinstance(tag, Tag)
        warnings = self.flushWarnings(offendingFunctions=[self.loaderFactory])
        if self.deprecatedUse:
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]['category'], DeprecationWarning)
            self.assertEqual(warnings[0]['message'], 'Passing filenames or file objects to XMLFile is deprecated since Twisted 12.1.  Pass a FilePath instead.')
        else:
            self.assertEqual(len(warnings), 0)
        self.assertEqual(tag.tagName, 'p')
        self.assertEqual(tag.children, ['Hello, world.'])

    def test_loadTwice(self) -> None:
        """
        If {load()} can be called on a loader twice the result should be the
        same.
        """
        assert isinstance(self, TestCase)
        loader = self.loaderFactory()
        tags1 = loader.load()
        tags2 = loader.load()
        self.assertEqual(tags1, tags2)
    test_loadTwice.suppress = [_xmlFileSuppress]