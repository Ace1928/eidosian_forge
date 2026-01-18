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
class RenderElementTests(TestCase):
    """
    Test L{renderElement}
    """

    def setUp(self) -> None:
        """
        Set up a common L{DummyRenderRequest}.
        """
        self.request = DummyRenderRequest()

    def test_simpleRender(self) -> Deferred[None]:
        """
        L{renderElement} returns NOT_DONE_YET and eventually
        writes the rendered L{Element} to the request before finishing the
        request.
        """
        element = TestElement()
        d = self.request.notifyFinish()

        def check(_: object) -> None:
            self.assertEqual(b''.join(self.request.written), b'<!DOCTYPE html>\n<p>Hello, world.</p>')
            self.assertTrue(self.request.finished)
        d.addCallback(check)
        self.assertIdentical(NOT_DONE_YET, renderElement(self.request, element))
        return d

    def test_simpleFailure(self) -> Deferred[None]:
        """
        L{renderElement} handles failures by writing a minimal
        error message to the request and finishing it.
        """
        element = FailingElement()
        d = self.request.notifyFinish()

        def check(_: object) -> None:
            flushed = self.flushLoggedErrors(FlattenerError)
            self.assertEqual(len(flushed), 1)
            self.assertEqual(b''.join(self.request.written), b'<!DOCTYPE html>\n<div style="font-size:800%;background-color:#FFF;color:#F00">An error occurred while rendering the response.</div>')
            self.assertTrue(self.request.finished)
        d.addCallback(check)
        self.assertIdentical(NOT_DONE_YET, renderElement(self.request, element))
        return d

    def test_simpleFailureWithTraceback(self) -> Deferred[None]:
        """
        L{renderElement} will render a traceback when rendering of
        the element fails and our site is configured to display tracebacks.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        self.request.site.displayTracebacks = True
        element = FailingElement()
        d = self.request.notifyFinish()

        def check(_: object) -> None:
            self.assertEquals(1, len(logObserver))
            f = logObserver[0]['log_failure']
            self.assertIsInstance(f.value, FlattenerError)
            flushed = self.flushLoggedErrors(FlattenerError)
            self.assertEqual(len(flushed), 1)
            self.assertEqual(b''.join(self.request.written), b'<!DOCTYPE html>\n<p>I failed.</p>')
            self.assertTrue(self.request.finished)
        d.addCallback(check)
        renderElement(self.request, element, _failElement=TestFailureElement)
        return d

    def test_nonDefaultDoctype(self) -> Deferred[None]:
        """
        L{renderElement} will write the doctype string specified by the
        doctype keyword argument.
        """
        element = TestElement()
        d = self.request.notifyFinish()

        def check(_: object) -> None:
            self.assertEqual(b''.join(self.request.written), b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n<p>Hello, world.</p>')
        d.addCallback(check)
        renderElement(self.request, element, doctype=b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">')
        return d

    def test_noneDoctype(self) -> Deferred[None]:
        """
        L{renderElement} will not write out a doctype if the doctype keyword
        argument is L{None}.
        """
        element = TestElement()
        d = self.request.notifyFinish()

        def check(_: object) -> None:
            self.assertEqual(b''.join(self.request.written), b'<p>Hello, world.</p>')
        d.addCallback(check)
        renderElement(self.request, element, doctype=None)
        return d