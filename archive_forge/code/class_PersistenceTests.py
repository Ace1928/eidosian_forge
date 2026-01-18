import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
class PersistenceTests(unittest.TestCase):
    """
    Tests for persistent HTTP connections.
    """

    def setUp(self):
        self.channel = http.HTTPChannel()
        self.request = _prequest()

    def test_http09(self):
        """
        After being used for an I{HTTP/0.9} request, the L{HTTPChannel} is not
        persistent.
        """
        persist = self.channel.checkPersistence(self.request, b'HTTP/0.9')
        self.assertFalse(persist)
        self.assertEqual([], list(self.request.responseHeaders.getAllRawHeaders()))

    def test_http10(self):
        """
        After being used for an I{HTTP/1.0} request, the L{HTTPChannel} is not
        persistent.
        """
        persist = self.channel.checkPersistence(self.request, b'HTTP/1.0')
        self.assertFalse(persist)
        self.assertEqual([], list(self.request.responseHeaders.getAllRawHeaders()))

    def test_http11(self):
        """
        After being used for an I{HTTP/1.1} request, the L{HTTPChannel} is
        persistent.
        """
        persist = self.channel.checkPersistence(self.request, b'HTTP/1.1')
        self.assertTrue(persist)
        self.assertEqual([], list(self.request.responseHeaders.getAllRawHeaders()))

    def test_http11Close(self):
        """
        After being used for an I{HTTP/1.1} request with a I{Connection: Close}
        header, the L{HTTPChannel} is not persistent.
        """
        request = _prequest(connection=[b'close'])
        persist = self.channel.checkPersistence(request, b'HTTP/1.1')
        self.assertFalse(persist)
        self.assertEqual([(b'Connection', [b'close'])], list(request.responseHeaders.getAllRawHeaders()))