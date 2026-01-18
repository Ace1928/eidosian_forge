import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
class StaticProducerTests(TestCase):
    """
    Tests for the abstract L{StaticProducer}.
    """

    def test_stopProducingClosesFile(self):
        """
        L{StaticProducer.stopProducing} closes the file object the producer is
        producing data from.
        """
        fileObject = StringIO()
        producer = static.StaticProducer(None, fileObject)
        producer.stopProducing()
        self.assertTrue(fileObject.closed)

    def test_stopProducingSetsRequestToNone(self):
        """
        L{StaticProducer.stopProducing} sets the request instance variable to
        None, which indicates to subclasses' resumeProducing methods that no
        more data should be produced.
        """
        fileObject = StringIO()
        producer = static.StaticProducer(DummyRequest([]), fileObject)
        producer.stopProducing()
        self.assertIdentical(None, producer.request)