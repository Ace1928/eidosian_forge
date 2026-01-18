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
class EnvironTests(WSGITestsMixin, TestCase):
    """
    Tests for the values in the C{environ} C{dict} passed to the application
    object by L{twisted.web.wsgi.WSGIResource}.
    """

    def environKeyEqual(self, key, value):

        def assertEnvironKeyEqual(result):
            environ, startResponse = result
            self.assertEqual(environ[key], value)
            return value
        return assertEnvironKeyEqual

    def test_environIsDict(self):
        """
        L{WSGIResource} calls the application object with an C{environ}
        parameter which is exactly of type C{dict}.
        """
        d = self.render('GET', '1.1', [], [''])

        def cbRendered(result):
            environ, startResponse = result
            self.assertIdentical(type(environ), dict)
            for name in environ:
                self.assertIsInstance(name, str)
        d.addCallback(cbRendered)
        return d

    def test_requestMethod(self):
        """
        The C{'REQUEST_METHOD'} key of the C{environ} C{dict} passed to the
        application contains the HTTP method in the request (RFC 3875, section
        4.1.12).
        """
        get = self.render('GET', '1.1', [], [''])
        get.addCallback(self.environKeyEqual('REQUEST_METHOD', 'GET'))
        post = self.render('POST', '1.1', [], [''])
        post.addCallback(self.environKeyEqual('REQUEST_METHOD', 'POST'))
        return gatherResults([get, post])

    def test_requestMethodIsNativeString(self):
        """
        The C{'REQUEST_METHOD'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        for method in (b'GET', 'GET'):
            request, result = self.prepareRequest()
            request.requestReceived(method)
            result.addCallback(self.environKeyEqual('REQUEST_METHOD', 'GET'))
            self.assertIsInstance(self.successResultOf(result), str)

    def test_scriptName(self):
        """
        The C{'SCRIPT_NAME'} key of the C{environ} C{dict} passed to the
        application contains the I{abs_path} (RFC 2396, section 3) to this
        resource (RFC 3875, section 4.1.13).
        """
        root = self.render('GET', '1.1', [], [''])
        root.addCallback(self.environKeyEqual('SCRIPT_NAME', ''))
        emptyChild = self.render('GET', '1.1', [''], [''])
        emptyChild.addCallback(self.environKeyEqual('SCRIPT_NAME', '/'))
        leaf = self.render('GET', '1.1', ['foo'], ['foo'])
        leaf.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo'))
        container = self.render('GET', '1.1', ['foo', ''], ['foo', ''])
        container.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo/'))
        internal = self.render('GET', '1.1', ['foo'], ['foo', 'bar'])
        internal.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo'))
        unencoded = self.render('GET', '1.1', ['foo', '/', b'bar\xff'], ['foo', '/', b'bar\xff'])
        unencoded.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo///barÿ'))
        return gatherResults([root, emptyChild, leaf, container, internal, unencoded])

    def test_scriptNameIsNativeString(self):
        """
        The C{'SCRIPT_NAME'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        request, result = self.prepareRequest()
        request.requestReceived(path=b'/res')
        result.addCallback(self.environKeyEqual('SCRIPT_NAME', '/res'))
        self.assertIsInstance(self.successResultOf(result), str)
        request, result = self.prepareRequest()
        self.assertRaises(TypeError, request.requestReceived, path='/res')

    def test_pathInfo(self):
        """
        The C{'PATH_INFO'} key of the C{environ} C{dict} passed to the
        application contains the suffix of the request URI path which is not
        included in the value for the C{'SCRIPT_NAME'} key (RFC 3875, section
        4.1.5).
        """
        assertKeyEmpty = self.environKeyEqual('PATH_INFO', '')
        root = self.render('GET', '1.1', [], [''])
        root.addCallback(self.environKeyEqual('PATH_INFO', '/'))
        emptyChild = self.render('GET', '1.1', [''], [''])
        emptyChild.addCallback(assertKeyEmpty)
        leaf = self.render('GET', '1.1', ['foo'], ['foo'])
        leaf.addCallback(assertKeyEmpty)
        container = self.render('GET', '1.1', ['foo', ''], ['foo', ''])
        container.addCallback(assertKeyEmpty)
        internalLeaf = self.render('GET', '1.1', ['foo'], ['foo', 'bar'])
        internalLeaf.addCallback(self.environKeyEqual('PATH_INFO', '/bar'))
        internalContainer = self.render('GET', '1.1', ['foo'], ['foo', ''])
        internalContainer.addCallback(self.environKeyEqual('PATH_INFO', '/'))
        unencoded = self.render('GET', '1.1', [], ['foo', '/', b'bar\xff'])
        unencoded.addCallback(self.environKeyEqual('PATH_INFO', '/foo///barÿ'))
        return gatherResults([root, leaf, container, internalLeaf, internalContainer, unencoded])

    def test_pathInfoIsNativeString(self):
        """
        The C{'PATH_INFO'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        request, result = self.prepareRequest()
        request.requestReceived(path=b'/res/foo/bar')
        result.addCallback(self.environKeyEqual('PATH_INFO', '/foo/bar'))
        self.assertIsInstance(self.successResultOf(result), str)
        request, result = self.prepareRequest()
        self.assertRaises(TypeError, request.requestReceived, path='/res/foo/bar')

    def test_queryString(self):
        """
        The C{'QUERY_STRING'} key of the C{environ} C{dict} passed to the
        application contains the portion of the request URI after the first
        I{?} (RFC 3875, section 4.1.7).
        """
        missing = self.render('GET', '1.1', [], [''], None)
        missing.addCallback(self.environKeyEqual('QUERY_STRING', ''))
        empty = self.render('GET', '1.1', [], [''], [])
        empty.addCallback(self.environKeyEqual('QUERY_STRING', ''))
        present = self.render('GET', '1.1', [], [''], [('foo', 'bar')])
        present.addCallback(self.environKeyEqual('QUERY_STRING', 'foo=bar'))
        unencoded = self.render('GET', '1.1', [], [''], [('/', '/')])
        unencoded.addCallback(self.environKeyEqual('QUERY_STRING', '%2F=%2F'))
        doubleQuestion = self.render('GET', '1.1', [], [''], [('foo', '?bar')], safe='?')
        doubleQuestion.addCallback(self.environKeyEqual('QUERY_STRING', 'foo=?bar'))
        return gatherResults([missing, empty, present, unencoded, doubleQuestion])

    def test_queryStringIsNativeString(self):
        """
        The C{'QUERY_STRING'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        request, result = self.prepareRequest()
        request.requestReceived(path=b'/res?foo=bar')
        result.addCallback(self.environKeyEqual('QUERY_STRING', 'foo=bar'))
        self.assertIsInstance(self.successResultOf(result), str)
        request, result = self.prepareRequest()
        self.assertRaises(TypeError, request.requestReceived, path='/res?foo=bar')

    def test_contentType(self):
        """
        The C{'CONTENT_TYPE'} key of the C{environ} C{dict} passed to the
        application contains the value of the I{Content-Type} request header
        (RFC 3875, section 4.1.3).
        """
        missing = self.render('GET', '1.1', [], [''])
        missing.addCallback(self.environKeyEqual('CONTENT_TYPE', ''))
        present = self.render('GET', '1.1', [], [''], None, [('content-type', 'x-foo/bar')])
        present.addCallback(self.environKeyEqual('CONTENT_TYPE', 'x-foo/bar'))
        return gatherResults([missing, present])

    def test_contentTypeIsNativeString(self):
        """
        The C{'CONTENT_TYPE'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        for contentType in (b'x-foo/bar', 'x-foo/bar'):
            request, result = self.prepareRequest()
            request.requestHeaders.addRawHeader(b'Content-Type', contentType)
            request.requestReceived()
            result.addCallback(self.environKeyEqual('CONTENT_TYPE', 'x-foo/bar'))
            self.assertIsInstance(self.successResultOf(result), str)

    def test_contentLength(self):
        """
        The C{'CONTENT_LENGTH'} key of the C{environ} C{dict} passed to the
        application contains the value of the I{Content-Length} request header
        (RFC 3875, section 4.1.2).
        """
        missing = self.render('GET', '1.1', [], [''])
        missing.addCallback(self.environKeyEqual('CONTENT_LENGTH', ''))
        present = self.render('GET', '1.1', [], [''], None, [('content-length', '1234')])
        present.addCallback(self.environKeyEqual('CONTENT_LENGTH', '1234'))
        return gatherResults([missing, present])

    def test_contentLengthIsNativeString(self):
        """
        The C{'CONTENT_LENGTH'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        for contentLength in (b'1234', '1234'):
            request, result = self.prepareRequest()
            request.requestHeaders.addRawHeader(b'Content-Length', contentLength)
            request.requestReceived()
            result.addCallback(self.environKeyEqual('CONTENT_LENGTH', '1234'))
            self.assertIsInstance(self.successResultOf(result), str)

    def test_serverName(self):
        """
        The C{'SERVER_NAME'} key of the C{environ} C{dict} passed to the
        application contains the best determination of the server hostname
        possible, using either the value of the I{Host} header in the request
        or the address the server is listening on if that header is not
        present (RFC 3875, section 4.1.14).
        """
        missing = self.render('GET', '1.1', [], [''])
        missing.addCallback(self.environKeyEqual('SERVER_NAME', '10.0.0.1'))
        present = self.render('GET', '1.1', [], [''], None, [('host', 'example.org')])
        present.addCallback(self.environKeyEqual('SERVER_NAME', 'example.org'))
        return gatherResults([missing, present])

    def test_serverNameIsNativeString(self):
        """
        The C{'SERVER_NAME'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        for serverName in (b'host.example.com', 'host.example.com'):
            request, result = self.prepareRequest()
            request.getRequestHostname = lambda: serverName
            request.requestReceived()
            result.addCallback(self.environKeyEqual('SERVER_NAME', 'host.example.com'))
            self.assertIsInstance(self.successResultOf(result), str)

    def test_serverPort(self):
        """
        The C{'SERVER_PORT'} key of the C{environ} C{dict} passed to the
        application contains the port number of the server which received the
        request (RFC 3875, section 4.1.15).
        """
        portNumber = 12354

        def makeChannel():
            channel = DummyChannel()
            channel.transport = DummyChannel.TCP()
            channel.transport.port = portNumber
            return channel
        self.channelFactory = makeChannel
        d = self.render('GET', '1.1', [], [''])
        d.addCallback(self.environKeyEqual('SERVER_PORT', str(portNumber)))
        return d

    def test_serverPortIsNativeString(self):
        """
        The C{'SERVER_PORT'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        request, result = self.prepareRequest()
        request.requestReceived()
        result.addCallback(self.environKeyEqual('SERVER_PORT', '80'))
        self.assertIsInstance(self.successResultOf(result), str)

    def test_serverProtocol(self):
        """
        The C{'SERVER_PROTOCOL'} key of the C{environ} C{dict} passed to the
        application contains the HTTP version number received in the request
        (RFC 3875, section 4.1.16).
        """
        old = self.render('GET', '1.0', [], [''])
        old.addCallback(self.environKeyEqual('SERVER_PROTOCOL', 'HTTP/1.0'))
        new = self.render('GET', '1.1', [], [''])
        new.addCallback(self.environKeyEqual('SERVER_PROTOCOL', 'HTTP/1.1'))
        return gatherResults([old, new])

    def test_serverProtocolIsNativeString(self):
        """
        The C{'SERVER_PROTOCOL'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
        for serverProtocol in (b'1.1', '1.1'):
            request, result = self.prepareRequest()
            request.write = lambda data: None
            request.requestReceived(version=b'1.1')
            result.addCallback(self.environKeyEqual('SERVER_PROTOCOL', '1.1'))
            self.assertIsInstance(self.successResultOf(result), str)

    def test_remoteAddr(self):
        """
        The C{'REMOTE_ADDR'} key of the C{environ} C{dict} passed to the
        application contains the address of the client making the request.
        """
        d = self.render('GET', '1.1', [], [''])
        d.addCallback(self.environKeyEqual('REMOTE_ADDR', '192.168.1.1'))
        return d

    def test_remoteAddrIPv6(self):
        """
        The C{'REMOTE_ADDR'} key of the C{environ} C{dict} passed to
        the application contains the address of the client making the
        request when connecting over IPv6.
        """

        def channelFactory():
            return DummyChannel(peer=IPv6Address('TCP', '::1', 1234))
        d = self.render('GET', '1.1', [], [''], channelFactory=channelFactory)
        d.addCallback(self.environKeyEqual('REMOTE_ADDR', '::1'))
        return d

    def test_headers(self):
        """
        HTTP request headers are copied into the C{environ} C{dict} passed to
        the application with a C{HTTP_} prefix added to their names.
        """
        singleValue = self.render('GET', '1.1', [], [''], None, [('foo', 'bar'), ('baz', 'quux')])

        def cbRendered(result):
            environ, startResponse = result
            self.assertEqual(environ['HTTP_FOO'], 'bar')
            self.assertEqual(environ['HTTP_BAZ'], 'quux')
        singleValue.addCallback(cbRendered)
        multiValue = self.render('GET', '1.1', [], [''], None, [('foo', 'bar'), ('foo', 'baz')])
        multiValue.addCallback(self.environKeyEqual('HTTP_FOO', 'bar,baz'))
        withHyphen = self.render('GET', '1.1', [], [''], None, [('foo-bar', 'baz')])
        withHyphen.addCallback(self.environKeyEqual('HTTP_FOO_BAR', 'baz'))
        multiLine = self.render('GET', '1.1', [], [''], None, [('foo', 'bar\n\tbaz')])
        multiLine.addCallback(self.environKeyEqual('HTTP_FOO', 'bar \tbaz'))
        return gatherResults([singleValue, multiValue, withHyphen, multiLine])

    def test_wsgiVersion(self):
        """
        The C{'wsgi.version'} key of the C{environ} C{dict} passed to the
        application has the value C{(1, 0)} indicating that this is a WSGI 1.0
        container.
        """
        versionDeferred = self.render('GET', '1.1', [], [''])
        versionDeferred.addCallback(self.environKeyEqual('wsgi.version', (1, 0)))
        return versionDeferred

    def test_wsgiRunOnce(self):
        """
        The C{'wsgi.run_once'} key of the C{environ} C{dict} passed to the
        application is set to C{False}.
        """
        once = self.render('GET', '1.1', [], [''])
        once.addCallback(self.environKeyEqual('wsgi.run_once', False))
        return once

    def test_wsgiMultithread(self):
        """
        The C{'wsgi.multithread'} key of the C{environ} C{dict} passed to the
        application is set to C{True}.
        """
        thread = self.render('GET', '1.1', [], [''])
        thread.addCallback(self.environKeyEqual('wsgi.multithread', True))
        return thread

    def test_wsgiMultiprocess(self):
        """
        The C{'wsgi.multiprocess'} key of the C{environ} C{dict} passed to the
        application is set to C{False}.
        """
        process = self.render('GET', '1.1', [], [''])
        process.addCallback(self.environKeyEqual('wsgi.multiprocess', False))
        return process

    def test_wsgiURLScheme(self):
        """
        The C{'wsgi.url_scheme'} key of the C{environ} C{dict} passed to the
        application has the request URL scheme.
        """

        def channelFactory():
            channel = DummyChannel()
            channel.transport = DummyChannel.SSL()
            return channel
        self.channelFactory = DummyChannel
        httpDeferred = self.render('GET', '1.1', [], [''])
        httpDeferred.addCallback(self.environKeyEqual('wsgi.url_scheme', 'http'))
        self.channelFactory = channelFactory
        httpsDeferred = self.render('GET', '1.1', [], [''])
        httpsDeferred.addCallback(self.environKeyEqual('wsgi.url_scheme', 'https'))
        return gatherResults([httpDeferred, httpsDeferred])

    def test_wsgiErrors(self):
        """
        The C{'wsgi.errors'} key of the C{environ} C{dict} passed to the
        application is a file-like object (as defined in the U{Input and Errors
        Streams<http://www.python.org/dev/peps/pep-0333/#input-and-error-streams>}
        section of PEP 333) which converts bytes written to it into events for
        the logging system.
        """
        events = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        errors = self.render('GET', '1.1', [], [''])

        def cbErrors(result):
            environ, startApplication = result
            errors = environ['wsgi.errors']
            errors.write('some message\n')
            errors.writelines(['another\nmessage\n'])
            errors.flush()
            self.assertEqual(events[0]['message'], ('some message\n',))
            self.assertEqual(events[0]['system'], 'wsgi')
            self.assertTrue(events[0]['isError'])
            self.assertEqual(events[1]['message'], ('another\nmessage\n',))
            self.assertEqual(events[1]['system'], 'wsgi')
            self.assertTrue(events[1]['isError'])
            self.assertEqual(len(events), 2)
        errors.addCallback(cbErrors)
        return errors

    def test_wsgiErrorsAcceptsOnlyNativeStringsInPython3(self):
        """
        The C{'wsgi.errors'} file-like object from the C{environ} C{dict}
        permits writes of only native strings in Python 3, and raises
        C{TypeError} for writes of non-native strings.
        """
        request, result = self.prepareRequest()
        request.requestReceived()
        environ, _ = self.successResultOf(result)
        errors = environ['wsgi.errors']
        error = self.assertRaises(TypeError, errors.write, b'fred')
        self.assertEqual("write() argument must be str, not b'fred' (bytes)", str(error))