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
class StaticMakeProducerTests(TestCase):
    """
    Tests for L{File.makeProducer}.
    """

    def makeResourceWithContent(self, content, type=None, encoding=None):
        """
        Make a L{static.File} resource that has C{content} for its content.

        @param content: The L{bytes} to use as the contents of the resource.
        @param type: Optional value for the content type of the resource.
        """
        fileName = FilePath(self.mktemp())
        fileName.setContent(content)
        resource = static.File(fileName._asBytesPath())
        resource.encoding = encoding
        resource.type = type
        return resource

    def contentHeaders(self, request):
        """
        Extract the content-* headers from the L{DummyRequest} C{request}.

        This returns the subset of C{request.outgoingHeaders} of headers that
        start with 'content-'.
        """
        contentHeaders = {}
        for k, v in request.responseHeaders.getAllRawHeaders():
            if k.lower().startswith(b'content-'):
                contentHeaders[k.lower()] = v[0]
        return contentHeaders

    def test_noRangeHeaderGivesNoRangeStaticProducer(self):
        """
        makeProducer when no Range header is set returns an instance of
        NoRangeStaticProducer.
        """
        resource = self.makeResourceWithContent(b'')
        request = DummyRequest([])
        with resource.openForReading() as file:
            producer = resource.makeProducer(request, file)
            self.assertIsInstance(producer, static.NoRangeStaticProducer)

    def test_noRangeHeaderSets200OK(self):
        """
        makeProducer when no Range header is set sets the responseCode on the
        request to 'OK'.
        """
        resource = self.makeResourceWithContent(b'')
        request = DummyRequest([])
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual(http.OK, request.responseCode)

    def test_noRangeHeaderSetsContentHeaders(self):
        """
        makeProducer when no Range header is set sets the Content-* headers
        for the response.
        """
        length = 123
        contentType = 'text/plain'
        contentEncoding = 'gzip'
        resource = self.makeResourceWithContent(b'a' * length, type=contentType, encoding=contentEncoding)
        request = DummyRequest([])
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual({b'content-type': networkString(contentType), b'content-length': b'%d' % (length,), b'content-encoding': networkString(contentEncoding)}, self.contentHeaders(request))

    def test_singleRangeGivesSingleRangeStaticProducer(self):
        """
        makeProducer when the Range header requests a single byte range
        returns an instance of SingleRangeStaticProducer.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3')
        resource = self.makeResourceWithContent(b'abcdef')
        with resource.openForReading() as file:
            producer = resource.makeProducer(request, file)
            self.assertIsInstance(producer, static.SingleRangeStaticProducer)

    def test_singleRangeSets206PartialContent(self):
        """
        makeProducer when the Range header requests a single, satisfiable byte
        range sets the response code on the request to 'Partial Content'.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3')
        resource = self.makeResourceWithContent(b'abcdef')
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual(http.PARTIAL_CONTENT, request.responseCode)

    def test_singleRangeSetsContentHeaders(self):
        """
        makeProducer when the Range header requests a single, satisfiable byte
        range sets the Content-* headers appropriately.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3')
        contentType = 'text/plain'
        contentEncoding = 'gzip'
        resource = self.makeResourceWithContent(b'abcdef', type=contentType, encoding=contentEncoding)
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual({b'content-type': networkString(contentType), b'content-encoding': networkString(contentEncoding), b'content-range': b'bytes 1-3/6', b'content-length': b'3'}, self.contentHeaders(request))

    def test_singleUnsatisfiableRangeReturnsSingleRangeStaticProducer(self):
        """
        makeProducer still returns an instance of L{SingleRangeStaticProducer}
        when the Range header requests a single unsatisfiable byte range.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=4-10')
        resource = self.makeResourceWithContent(b'abc')
        with resource.openForReading() as file:
            producer = resource.makeProducer(request, file)
            self.assertIsInstance(producer, static.SingleRangeStaticProducer)

    def test_singleUnsatisfiableRangeSets416ReqestedRangeNotSatisfiable(self):
        """
        makeProducer sets the response code of the request to of 'Requested
        Range Not Satisfiable' when the Range header requests a single
        unsatisfiable byte range.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=4-10')
        resource = self.makeResourceWithContent(b'abc')
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual(http.REQUESTED_RANGE_NOT_SATISFIABLE, request.responseCode)

    def test_singleUnsatisfiableRangeSetsContentHeaders(self):
        """
        makeProducer when the Range header requests a single, unsatisfiable
        byte range sets the Content-* headers appropriately.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=4-10')
        contentType = 'text/plain'
        resource = self.makeResourceWithContent(b'abc', type=contentType)
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual({b'content-type': b'text/plain', b'content-length': b'0', b'content-range': b'bytes */3'}, self.contentHeaders(request))

    def test_singlePartiallyOverlappingRangeSetsContentHeaders(self):
        """
        makeProducer when the Range header requests a single byte range that
        partly overlaps the resource sets the Content-* headers appropriately.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=2-10')
        contentType = 'text/plain'
        resource = self.makeResourceWithContent(b'abc', type=contentType)
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual({b'content-type': b'text/plain', b'content-length': b'1', b'content-range': b'bytes 2-2/3'}, self.contentHeaders(request))

    def test_multipleRangeGivesMultipleRangeStaticProducer(self):
        """
        makeProducer when the Range header requests a single byte range
        returns an instance of MultipleRangeStaticProducer.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3,5-6')
        resource = self.makeResourceWithContent(b'abcdef')
        with resource.openForReading() as file:
            producer = resource.makeProducer(request, file)
            self.assertIsInstance(producer, static.MultipleRangeStaticProducer)

    def test_multipleRangeSets206PartialContent(self):
        """
        makeProducer when the Range header requests a multiple satisfiable
        byte ranges sets the response code on the request to 'Partial
        Content'.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3,5-6')
        resource = self.makeResourceWithContent(b'abcdef')
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual(http.PARTIAL_CONTENT, request.responseCode)

    def test_mutipleRangeSetsContentHeaders(self):
        """
        makeProducer when the Range header requests a single, satisfiable byte
        range sets the Content-* headers appropriately.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3,5-6')
        resource = self.makeResourceWithContent(b'abcdefghijkl', encoding='gzip')
        with resource.openForReading() as file:
            producer = resource.makeProducer(request, file)
            contentHeaders = self.contentHeaders(request)
            self.assertEqual({b'content-length', b'content-type'}, set(contentHeaders.keys()))
            expectedLength = 5
            for boundary, offset, size in producer.rangeInfo:
                expectedLength += len(boundary)
            self.assertEqual(b'%d' % (expectedLength,), contentHeaders[b'content-length'])
            self.assertIn(b'content-type', contentHeaders)
            contentType = contentHeaders[b'content-type']
            self.assertNotIdentical(None, re.match(b'multipart/byteranges; boundary="[^"]*"\\Z', contentType))
            self.assertNotIn(b'content-encoding', contentHeaders)

    def test_multipleUnsatisfiableRangesReturnsMultipleRangeStaticProducer(self):
        """
        makeProducer still returns an instance of L{SingleRangeStaticProducer}
        when the Range header requests multiple ranges, none of which are
        satisfiable.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=10-12,15-20')
        resource = self.makeResourceWithContent(b'abc')
        with resource.openForReading() as file:
            producer = resource.makeProducer(request, file)
            self.assertIsInstance(producer, static.MultipleRangeStaticProducer)

    def test_multipleUnsatisfiableRangesSets416ReqestedRangeNotSatisfiable(self):
        """
        makeProducer sets the response code of the request to of 'Requested
        Range Not Satisfiable' when the Range header requests multiple ranges,
        none of which are satisfiable.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=10-12,15-20')
        resource = self.makeResourceWithContent(b'abc')
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual(http.REQUESTED_RANGE_NOT_SATISFIABLE, request.responseCode)

    def test_multipleUnsatisfiableRangeSetsContentHeaders(self):
        """
        makeProducer when the Range header requests multiple ranges, none of
        which are satisfiable, sets the Content-* headers appropriately.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=4-10')
        contentType = 'text/plain'
        request.requestHeaders.addRawHeader(b'range', b'bytes=10-12,15-20')
        resource = self.makeResourceWithContent(b'abc', type=contentType)
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual({b'content-length': b'0', b'content-range': b'bytes */3', b'content-type': b'text/plain'}, self.contentHeaders(request))

    def test_oneSatisfiableRangeIsEnough(self):
        """
        makeProducer when the Range header requests multiple ranges, at least
        one of which matches, sets the response code to 'Partial Content'.
        """
        request = DummyRequest([])
        request.requestHeaders.addRawHeader(b'range', b'bytes=1-3,100-200')
        resource = self.makeResourceWithContent(b'abcdef')
        with resource.openForReading() as file:
            resource.makeProducer(request, file)
            self.assertEqual(http.PARTIAL_CONTENT, request.responseCode)