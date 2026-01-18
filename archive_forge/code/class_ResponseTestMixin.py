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
class ResponseTestMixin:
    """
    A mixin that provides a simple means of comparing an actual response string
    to an expected response string by performing the minimal parsing.
    """

    def assertResponseEquals(self, responses, expected):
        """
        Assert that the C{responses} matches the C{expected} responses.

        @type responses: C{bytes}
        @param responses: The bytes sent in response to one or more requests.

        @type expected: C{list} of C{tuple} of C{bytes}
        @param expected: The expected values for the responses.  Each tuple
            element of the list represents one response.  Each byte string
            element of the tuple is a full header line without delimiter, except
            for the last element which gives the full response body.
        """
        for response in expected:
            expectedHeaders, expectedContent = (response[:-1], response[-1])
            expectedStatus = expectedHeaders[0]
            expectedHeaders = expectedHeaders[1:]
            headers, rest = responses.split(b'\r\n\r\n', 1)
            headers = headers.splitlines()
            status = headers.pop(0)
            self.assertEqual(expectedStatus, status)
            self.assertEqual(set(headers), set(expectedHeaders))
            content = rest[:len(expectedContent)]
            responses = rest[len(expectedContent):]
            self.assertEqual(content, expectedContent)