import tempfile
import traceback
import warnings
from sys import exc_info
from urllib.parse import quote as urlquote
from zope.interface.verify import verifyObject
from twisted.internet import reactor
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import Logger, globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import TestCase
from twisted.web import http
from twisted.web.resource import IResource, Resource
from twisted.web.server import Request, Site, version
from twisted.web.test.test_web import DummyChannel
from twisted.web.wsgi import WSGIResource
class WSGITestsMixin:
    """
    @ivar channelFactory: A no-argument callable which will be invoked to
        create a new HTTP channel to associate with request objects.
    """
    channelFactory = DummyChannel

    def setUp(self):
        self.threadpool = SynchronousThreadPool()
        self.reactor = SynchronousReactorThreads()

    def lowLevelRender(self, requestFactory, applicationFactory, channelFactory, method, version, resourceSegments, requestSegments, query=None, headers=[], body=None, safe=''):
        """
        @param method: A C{str} giving the request method to use.

        @param version: A C{str} like C{'1.1'} giving the request version.

        @param resourceSegments: A C{list} of unencoded path segments which
            specifies the location in the resource hierarchy at which the
            L{WSGIResource} will be placed, eg C{['']} for I{/}, C{['foo',
            'bar', '']} for I{/foo/bar/}, etc.

        @param requestSegments: A C{list} of unencoded path segments giving the
            request URI.

        @param query: A C{list} of two-tuples of C{str} giving unencoded query
            argument keys and values.

        @param headers: A C{list} of two-tuples of C{str} giving request header
            names and corresponding values.

        @param safe: A C{str} giving the bytes which are to be considered
            I{safe} for inclusion in the request URI and not quoted.

        @return: A L{Deferred} which will be called back with a two-tuple of
            the arguments passed which would be passed to the WSGI application
            object for this configuration and request (ie, the environment and
            start_response callable).
        """

        def _toByteString(string):
            if isinstance(string, bytes):
                return string
            else:
                return string.encode('iso-8859-1')
        root = WSGIResource(self.reactor, self.threadpool, applicationFactory())
        resourceSegments.reverse()
        for seg in resourceSegments:
            tmp = Resource()
            tmp.putChild(_toByteString(seg), root)
            root = tmp
        channel = channelFactory()
        channel.site = Site(root)
        request = requestFactory(channel, False)
        for k, v in headers:
            request.requestHeaders.addRawHeader(_toByteString(k), _toByteString(v))
        request.gotLength(0)
        if body:
            request.content.write(body)
            request.content.seek(0)
        uri = '/' + '/'.join([urlquote(seg, safe) for seg in requestSegments])
        if query is not None:
            uri += '?' + '&'.join(['='.join([urlquote(k, safe), urlquote(v, safe)]) for k, v in query])
        request.requestReceived(_toByteString(method), _toByteString(uri), b'HTTP/' + _toByteString(version))
        return request

    def render(self, *a, **kw):
        result = Deferred()

        def applicationFactory():

            def application(*args):
                environ, startResponse = args
                result.callback(args)
                startResponse('200 OK', [])
                return iter(())
            return application
        channelFactory = kw.pop('channelFactory', self.channelFactory)
        self.lowLevelRender(Request, applicationFactory, channelFactory, *a, **kw)
        return result

    def requestFactoryFactory(self, requestClass=Request):
        d = Deferred()

        def requestFactory(*a, **kw):
            request = requestClass(*a, **kw)
            request.notifyFinish().chainDeferred(d)
            return request
        return (d, requestFactory)

    def getContentFromResponse(self, response):
        return response.split(b'\r\n\r\n', 1)[1]

    def prepareRequest(self, application=None):
        """
        Prepare a L{Request} which, when a request is received, captures the
        C{environ} and C{start_response} callable passed to a WSGI app.

        @param application: An optional WSGI application callable that accepts
            the familiar C{environ} and C{start_response} args and returns an
            iterable of body content. If not supplied, C{start_response} will
            be called with a "200 OK" status and no headers, and no content
            will be yielded.

        @return: A two-tuple of (C{request}, C{deferred}). The former is a
            Twisted L{Request}. The latter is a L{Deferred} which will be
            called back with a two-tuple of the arguments passed to a WSGI
            application (i.e. the C{environ} and C{start_response} callable),
            or will errback with any error arising within the WSGI app.
        """
        result = Deferred()

        def outerApplication(environ, startResponse):
            try:
                if application is None:
                    startResponse('200 OK', [])
                    content = iter(())
                else:
                    content = application(environ, startResponse)
            except BaseException:
                result.errback()
                startResponse('500 Error', [])
                return iter(())
            else:
                result.callback((environ, startResponse))
                return content
        resource = WSGIResource(self.reactor, self.threadpool, outerApplication)
        root = Resource()
        root.putChild(b'res', resource)
        channel = self.channelFactory()
        channel.site = Site(root)

        class CannedRequest(Request):
            """
            Convenient L{Request} derivative which has canned values for all
            of C{requestReceived}'s arguments.
            """

            def requestReceived(self, command=b'GET', path=b'/res', version=b'1.1'):
                return Request.requestReceived(self, command=command, path=path, version=version)
        request = CannedRequest(channel, queued=False)
        request.gotLength(0)
        return (request, result)