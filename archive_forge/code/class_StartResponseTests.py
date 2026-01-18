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
class StartResponseTests(WSGITestsMixin, TestCase):
    """
    Tests for the I{start_response} parameter passed to the application object
    by L{WSGIResource}.
    """

    def test_status(self):
        """
        The response status passed to the I{start_response} callable is written
        as the status of the response to the request.
        """
        channel = DummyChannel()

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('107 Strange message', [])
                return iter(())
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertTrue(channel.transport.written.getvalue().startswith(b'HTTP/1.1 107 Strange message'))
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_statusMustBeNativeString(self):
        """
        The response status passed to the I{start_response} callable MUST be a
        native string in Python 2 and Python 3.
        """
        status = b'200 OK'

        def application(environ, startResponse):
            startResponse(status, [])
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertEqual("status must be str, not b'200 OK' (bytes)", str(error))
        return self.assertFailure(result, TypeError).addCallback(checkMessage)

    def _headersTest(self, appHeaders, expectedHeaders):
        """
        Verify that if the response headers given by C{appHeaders} are passed
        to the I{start_response} callable, then the response header lines given
        by C{expectedHeaders} plus I{Server} and I{Date} header lines are
        included in the response.
        """
        self.patch(http, 'datetimeToString', lambda: 'Tuesday')
        channel = DummyChannel()

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('200 OK', appHeaders)
                return iter(())
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            response = channel.transport.written.getvalue()
            headers, rest = response.split(b'\r\n\r\n', 1)
            headerLines = headers.split(b'\r\n')[1:]
            headerLines.sort()
            allExpectedHeaders = expectedHeaders + [b'Date: Tuesday', b'Server: ' + version, b'Transfer-Encoding: chunked']
            allExpectedHeaders.sort()
            self.assertEqual(headerLines, allExpectedHeaders)
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_headers(self):
        """
        The headers passed to the I{start_response} callable are included in
        the response as are the required I{Date} and I{Server} headers and the
        necessary connection (hop to hop) header I{Transfer-Encoding}.
        """
        return self._headersTest([('foo', 'bar'), ('baz', 'quux')], [b'Baz: quux', b'Foo: bar'])

    def test_headersMustBeSequence(self):
        """
        The headers passed to the I{start_response} callable MUST be a
        sequence.
        """
        headers = [('key', 'value')]

        def application(environ, startResponse):
            startResponse('200 OK', iter(headers))
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertRegex(str(error), 'headers must be a list, not <(list_?|sequence)iterator .+> [(]\\1iterator[)]')
        return self.assertFailure(result, TypeError).addCallback(checkMessage)

    def test_headersShouldBePlainList(self):
        """
        According to PEP-3333, the headers passed to the I{start_response}
        callable MUST be a plain list:

          The response_headers argument ... must be a Python list; i.e.
          type(response_headers) is ListType

        However, for bug-compatibility, any sequence is accepted. In both
        Python 2 and Python 3, only a warning is issued when a sequence other
        than a list is encountered.
        """

        def application(environ, startResponse):
            startResponse('200 OK', (('not', 'list'),))
            return iter(())
        request, result = self.prepareRequest(application)
        with warnings.catch_warnings(record=True) as caught:
            request.requestReceived()
            result = self.successResultOf(result)
        self.assertEqual(1, len(caught))
        self.assertEqual(RuntimeWarning, caught[0].category)
        self.assertEqual("headers should be a list, not (('not', 'list'),) (tuple)", str(caught[0].message))

    def test_headersMustEachBeSequence(self):
        """
        Each header passed to the I{start_response} callable MUST be a
        sequence.
        """
        header = ('key', 'value')

        def application(environ, startResponse):
            startResponse('200 OK', [iter(header)])
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertRegex(str(error), 'header must be a [(]str, str[)] tuple, not <(tuple_?|sequence)iterator .+> [(]\\1iterator[)]')
        return self.assertFailure(result, TypeError).addCallback(checkMessage)

    def test_headersShouldEachBeTuple(self):
        """
        According to PEP-3333, each header passed to the I{start_response}
        callable should be a tuple:

          The response_headers argument is a list of (header_name,
          header_value) tuples

        However, for bug-compatibility, any 2 element sequence is also
        accepted. In both Python 2 and Python 3, only a warning is issued when
        a sequence other than a tuple is encountered.
        """

        def application(environ, startResponse):
            startResponse('200 OK', [['not', 'tuple']])
            return iter(())
        request, result = self.prepareRequest(application)
        with warnings.catch_warnings(record=True) as caught:
            request.requestReceived()
            result = self.successResultOf(result)
        self.assertEqual(1, len(caught))
        self.assertEqual(RuntimeWarning, caught[0].category)
        self.assertEqual("header should be a (str, str) tuple, not ['not', 'tuple'] (list)", str(caught[0].message))

    def test_headersShouldEachHaveKeyAndValue(self):
        """
        Each header passed to the I{start_response} callable MUST hold a key
        and a value, and ONLY a key and a value.
        """

        def application(environ, startResponse):
            startResponse('200 OK', [('too', 'many', 'cooks')])
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertEqual("header must be a (str, str) tuple, not ('too', 'many', 'cooks')", str(error))
        return self.assertFailure(result, TypeError).addCallback(checkMessage)

    def test_headerKeyMustBeNativeString(self):
        """
        Each header key passed to the I{start_response} callable MUST be at
        native string in Python 2 and Python 3.
        """
        key = b'key'

        def application(environ, startResponse):
            startResponse('200 OK', [(key, 'value')])
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertEqual(f"header must be (str, str) tuple, not ({key!r}, 'value')", str(error))
        return self.assertFailure(result, TypeError).addCallback(checkMessage)

    def test_headerValueMustBeNativeString(self):
        """
        Each header value passed to the I{start_response} callable MUST be at
        native string in Python 2 and Python 3.
        """
        value = b'value'

        def application(environ, startResponse):
            startResponse('200 OK', [('key', value)])
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertEqual(f"header must be (str, str) tuple, not ('key', {value!r})", str(error))
        return self.assertFailure(result, TypeError).addCallback(checkMessage)

    def test_applicationProvidedContentType(self):
        """
        If I{Content-Type} is included in the headers passed to the
        I{start_response} callable, one I{Content-Type} header is included in
        the response.
        """
        return self._headersTest([('content-type', 'monkeys are great')], [b'Content-Type: monkeys are great'])

    def test_applicationProvidedServerAndDate(self):
        """
        If either I{Server} or I{Date} is included in the headers passed to the
        I{start_response} callable, they are disregarded.
        """
        return self._headersTest([('server', 'foo'), ('Server', 'foo'), ('date', 'bar'), ('dATE', 'bar')], [])

    def test_delayedUntilReturn(self):
        """
        Nothing is written in response to a request when the I{start_response}
        callable is invoked.  If the iterator returned by the application
        object produces only empty strings, the response is written after the
        last element is produced.
        """
        channel = DummyChannel()
        intermediateValues = []

        def record():
            intermediateValues.append(channel.transport.written.getvalue())

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('200 OK', [('foo', 'bar'), ('baz', 'quux')])
                yield b''
                record()
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertEqual(intermediateValues, [b''])
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_delayedUntilContent(self):
        """
        Nothing is written in response to a request when the I{start_response}
        callable is invoked.  Once a non-empty string has been produced by the
        iterator returned by the application object, the response status and
        headers are written.
        """
        channel = DummyChannel()
        intermediateValues = []

        def record():
            intermediateValues.append(channel.transport.written.getvalue())

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('200 OK', [('foo', 'bar')])
                yield b''
                record()
                yield b'foo'
                record()
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertFalse(intermediateValues[0])
            self.assertTrue(intermediateValues[1])
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_content(self):
        """
        Content produced by the iterator returned by the application object is
        written to the request as it is produced.
        """
        channel = DummyChannel()
        intermediateValues = []

        def record():
            intermediateValues.append(channel.transport.written.getvalue())

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('200 OK', [('content-length', '6')])
                yield b'foo'
                record()
                yield b'bar'
                record()
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertEqual(self.getContentFromResponse(intermediateValues[0]), b'foo')
            self.assertEqual(self.getContentFromResponse(intermediateValues[1]), b'foobar')
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_multipleStartResponse(self):
        """
        If the I{start_response} callable is invoked multiple times before a
        data for the response body is produced, the values from the last call
        are used.
        """
        channel = DummyChannel()

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('100 Foo', [])
                startResponse('200 Bar', [])
                return iter(())
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertTrue(channel.transport.written.getvalue().startswith(b'HTTP/1.1 200 Bar\r\n'))
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_startResponseWithException(self):
        """
        If the I{start_response} callable is invoked with a third positional
        argument before the status and headers have been written to the
        response, the status and headers become the newly supplied values.
        """
        channel = DummyChannel()

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('100 Foo', [], (Exception, Exception('foo'), None))
                return iter(())
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertTrue(channel.transport.written.getvalue().startswith(b'HTTP/1.1 100 Foo\r\n'))
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_startResponseWithExceptionTooLate(self):
        """
        If the I{start_response} callable is invoked with a third positional
        argument after the status and headers have been written to the
        response, the supplied I{exc_info} values are re-raised to the
        application.
        """
        channel = DummyChannel()

        class SomeException(Exception):
            pass
        try:
            raise SomeException()
        except BaseException:
            excInfo = exc_info()
        reraised = []

        def applicationFactory():

            def application(environ, startResponse):
                startResponse('200 OK', [])
                yield b'foo'
                try:
                    startResponse('500 ERR', [], excInfo)
                except BaseException:
                    reraised.append(exc_info())
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertTrue(channel.transport.written.getvalue().startswith(b'HTTP/1.1 200 OK\r\n'))
            self.assertEqual(reraised[0][0], excInfo[0])
            self.assertEqual(reraised[0][1], excInfo[1])
            tb1 = reraised[0][2].tb_next
            tb2 = excInfo[2]
            self.assertEqual(traceback.extract_tb(tb1)[1], traceback.extract_tb(tb2)[0])
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_write(self):
        """
        I{start_response} returns the I{write} callable which can be used to
        write bytes to the response body without buffering.
        """
        channel = DummyChannel()
        intermediateValues = []

        def record():
            intermediateValues.append(channel.transport.written.getvalue())

        def applicationFactory():

            def application(environ, startResponse):
                write = startResponse('100 Foo', [('content-length', '6')])
                write(b'foo')
                record()
                write(b'bar')
                record()
                return iter(())
            return application
        d, requestFactory = self.requestFactoryFactory()

        def cbRendered(ignored):
            self.assertEqual(self.getContentFromResponse(intermediateValues[0]), b'foo')
            self.assertEqual(self.getContentFromResponse(intermediateValues[1]), b'foobar')
        d.addCallback(cbRendered)
        self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
        return d

    def test_writeAcceptsOnlyByteStrings(self):
        """
        The C{write} callable returned from C{start_response} only accepts
        byte strings.
        """

        def application(environ, startResponse):
            write = startResponse('200 OK', [])
            write('bogus')
            return iter(())
        request, result = self.prepareRequest(application)
        request.requestReceived()

        def checkMessage(error):
            self.assertEqual("Can only write bytes to a transport, not 'bogus'", str(error))
        return self.assertFailure(result, TypeError).addCallback(checkMessage)