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
class StaticDataTests(TestCase):
    """
    Tests for L{Data}.
    """

    def test_headRequest(self):
        """
        L{Data.render} returns an empty response body for a I{HEAD} request.
        """
        data = static.Data(b'foo', 'bar')
        request = DummyRequest([''])
        request.method = b'HEAD'
        d = _render(data, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b'')
        d.addCallback(cbRendered)
        return d

    def test_invalidMethod(self):
        """
        L{Data.render} raises L{UnsupportedMethod} in response to a non-I{GET},
        non-I{HEAD} request.
        """
        data = static.Data(b'foo', b'bar')
        request = DummyRequest([b''])
        request.method = b'POST'
        self.assertRaises(UnsupportedMethod, data.render, request)