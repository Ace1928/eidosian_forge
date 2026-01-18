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
class GzipEncoderTests(unittest.TestCase):

    def setUp(self):
        self.channel = DummyChannel()
        staticResource = Data(b'Some data', 'text/plain')
        wrapped = resource.EncodingResourceWrapper(staticResource, [server.GzipEncoderFactory()])
        self.channel.site.resource.putChild(b'foo', wrapped)

    def test_interfaces(self):
        """
        L{server.GzipEncoderFactory} implements the
        L{iweb._IRequestEncoderFactory} and its C{encoderForRequest} returns an
        instance of L{server._GzipEncoder} which implements
        L{iweb._IRequestEncoder}.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'gzip,deflate'])
        factory = server.GzipEncoderFactory()
        self.assertTrue(verifyObject(iweb._IRequestEncoderFactory, factory))
        encoder = factory.encoderForRequest(request)
        self.assertTrue(verifyObject(iweb._IRequestEncoder, encoder))

    def test_encoding(self):
        """
        If the client request passes a I{Accept-Encoding} header which mentions
        gzip, L{server._GzipEncoder} automatically compresses the data.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'gzip,deflate'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_whitespaceInAcceptEncoding(self):
        """
        If the client request passes a I{Accept-Encoding} header which mentions
        gzip, with whitespace inbetween the encoding name and the commas,
        L{server._GzipEncoder} automatically compresses the data.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate, gzip'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_nonEncoding(self):
        """
        L{server.GzipEncoderFactory} doesn't return a L{server._GzipEncoder} if
        the I{Accept-Encoding} header doesn't mention gzip support.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'foo,bar'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertIn(b'Content-Length', data)
        self.assertNotIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', body)

    def test_multipleAccept(self):
        """
        If there are multiple I{Accept-Encoding} header,
        L{server.GzipEncoderFactory} reads them properly to detect if gzip is
        supported.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_alreadyEncoded(self):
        """
        If the content is already encoded and the I{Content-Encoding} header is
        set, L{server.GzipEncoderFactory} properly appends gzip to it.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
        request.responseHeaders.setRawHeaders(b'Content-Encoding', [b'deflate'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: deflate,gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_multipleEncodingLines(self):
        """
        If there are several I{Content-Encoding} headers,
        L{server.GzipEncoderFactory} normalizes it and appends gzip to the
        field value.
        """
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
        request.responseHeaders.setRawHeaders(b'Content-Encoding', [b'foo', b'bar'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: foo,bar,gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))