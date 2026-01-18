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
class DeprecatedRequestAttributesTests(unittest.TestCase):
    """
    Tests for deprecated attributes of L{twisted.web.http.Request}.
    """

    def test_getClientIP(self):
        """
        L{Request.getClientIP} is deprecated in favor of
        L{Request.getClientAddress}.
        """
        request = http.Request(DummyChannel(peer=address.IPv6Address('TCP', '127.0.0.1', 12345)))
        request.gotLength(0)
        request.requestReceived(b'GET', b'/', b'HTTP/1.1')
        request.getClientIP()
        warnings = self.flushWarnings(offendingFunctions=[self.test_getClientIP])
        self.assertEqual(1, len(warnings))
        self.assertEqual({'category': DeprecationWarning, 'message': 'twisted.web.http.Request.getClientIP was deprecated in Twisted 18.4.0; please use getClientAddress instead'}, sub(['category', 'message'], warnings[0]))

    def test_noLongerQueued(self):
        """
        L{Request.noLongerQueued} is deprecated, as we no longer process
        requests simultaneously.
        """
        channel = DummyChannel()
        request = http.Request(channel)
        request.noLongerQueued()
        warnings = self.flushWarnings(offendingFunctions=[self.test_noLongerQueued])
        self.assertEqual(1, len(warnings))
        self.assertEqual({'category': DeprecationWarning, 'message': 'twisted.web.http.Request.noLongerQueued was deprecated in Twisted 16.3.0'}, sub(['category', 'message'], warnings[0]))