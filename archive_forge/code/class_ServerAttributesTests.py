import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
class ServerAttributesTests(unittest.TestCase):
    """
    Tests that deprecated twisted.web.server attributes raise the appropriate
    deprecation warnings when used.
    """

    def test_deprecatedAttributeDateTimeString(self):
        """
        twisted.web.server.date_time_string should not be used; instead use
        twisted.web.http.datetimeToString directly
        """
        server.date_time_string
        warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedAttributeDateTimeString])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.web.server.date_time_string was deprecated in Twisted 12.1.0: Please use twisted.web.http.datetimeToString instead')

    def test_deprecatedAttributeStringDateTime(self):
        """
        twisted.web.server.string_date_time should not be used; instead use
        twisted.web.http.stringToDatetime directly
        """
        server.string_date_time
        warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedAttributeStringDateTime])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.web.server.string_date_time was deprecated in Twisted 12.1.0: Please use twisted.web.http.stringToDatetime instead')