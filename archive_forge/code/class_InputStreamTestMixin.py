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
class InputStreamTestMixin(WSGITestsMixin):
    """
    A mixin for L{TestCase} subclasses which defines a number of tests against
    L{_InputStream}.  The subclass is expected to create a file-like object to
    be wrapped by an L{_InputStream} under test.
    """

    def getFileType(self):
        raise NotImplementedError(f'{self.__class__.__name__}.getFile must be implemented')

    def _renderAndReturnReaderResult(self, reader, content):
        contentType = self.getFileType()

        class CustomizedRequest(Request):

            def gotLength(self, length):
                self.content = contentType()

        def appFactoryFactory(reader):
            result = Deferred()

            def applicationFactory():

                def application(*args):
                    environ, startResponse = args
                    result.callback(reader(environ['wsgi.input']))
                    startResponse('200 OK', [])
                    return iter(())
                return application
            return (result, applicationFactory)
        d, appFactory = appFactoryFactory(reader)
        self.lowLevelRender(CustomizedRequest, appFactory, DummyChannel, 'PUT', '1.1', [], [''], None, [], content)
        return d

    def test_readAll(self):
        """
        Calling L{_InputStream.read} with no arguments returns the entire input
        stream.
        """
        bytes = b'some bytes are here'
        d = self._renderAndReturnReaderResult(lambda input: input.read(), bytes)
        d.addCallback(self.assertEqual, bytes)
        return d

    def test_readSome(self):
        """
        Calling L{_InputStream.read} with an integer returns that many bytes
        from the input stream, as long as it is less than or equal to the total
        number of bytes available.
        """
        bytes = b'hello, world.'
        d = self._renderAndReturnReaderResult(lambda input: input.read(3), bytes)
        d.addCallback(self.assertEqual, b'hel')
        return d

    def test_readMoreThan(self):
        """
        Calling L{_InputStream.read} with an integer that is greater than the
        total number of bytes in the input stream returns all bytes in the
        input stream.
        """
        bytes = b'some bytes are here'
        d = self._renderAndReturnReaderResult(lambda input: input.read(len(bytes) + 3), bytes)
        d.addCallback(self.assertEqual, bytes)
        return d

    def test_readTwice(self):
        """
        Calling L{_InputStream.read} a second time returns bytes starting from
        the position after the last byte returned by the previous read.
        """
        bytes = b'some bytes, hello'

        def read(input):
            input.read(3)
            return input.read()
        d = self._renderAndReturnReaderResult(read, bytes)
        d.addCallback(self.assertEqual, bytes[3:])
        return d

    def test_readNone(self):
        """
        Calling L{_InputStream.read} with L{None} as an argument returns all
        bytes in the input stream.
        """
        bytes = b'the entire stream'
        d = self._renderAndReturnReaderResult(lambda input: input.read(None), bytes)
        d.addCallback(self.assertEqual, bytes)
        return d

    def test_readNegative(self):
        """
        Calling L{_InputStream.read} with a negative integer as an argument
        returns all bytes in the input stream.
        """
        bytes = b'all of the input'
        d = self._renderAndReturnReaderResult(lambda input: input.read(-1), bytes)
        d.addCallback(self.assertEqual, bytes)
        return d

    def test_readline(self):
        """
        Calling L{_InputStream.readline} with no argument returns one line from
        the input stream.
        """
        bytes = b'hello\nworld'
        d = self._renderAndReturnReaderResult(lambda input: input.readline(), bytes)
        d.addCallback(self.assertEqual, b'hello\n')
        return d

    def test_readlineSome(self):
        """
        Calling L{_InputStream.readline} with an integer returns at most that
        many bytes, even if it is not enough to make up a complete line.

        COMPATIBILITY NOTE: the size argument is excluded from the WSGI
        specification, but is provided here anyhow, because useful libraries
        such as python stdlib's cgi.py assume their input file-like-object
        supports readline with a size argument. If you use it, be aware your
        application may not be portable to other conformant WSGI servers.
        """
        bytes = b'goodbye\nworld'
        d = self._renderAndReturnReaderResult(lambda input: input.readline(3), bytes)
        d.addCallback(self.assertEqual, b'goo')
        return d

    def test_readlineMoreThan(self):
        """
        Calling L{_InputStream.readline} with an integer which is greater than
        the number of bytes in the next line returns only the next line.
        """
        bytes = b'some lines\nof text'
        d = self._renderAndReturnReaderResult(lambda input: input.readline(20), bytes)
        d.addCallback(self.assertEqual, b'some lines\n')
        return d

    def test_readlineTwice(self):
        """
        Calling L{_InputStream.readline} a second time returns the line
        following the line returned by the first call.
        """
        bytes = b'first line\nsecond line\nlast line'

        def readline(input):
            input.readline()
            return input.readline()
        d = self._renderAndReturnReaderResult(readline, bytes)
        d.addCallback(self.assertEqual, b'second line\n')
        return d

    def test_readlineNone(self):
        """
        Calling L{_InputStream.readline} with L{None} as an argument returns
        one line from the input stream.
        """
        bytes = b'this is one line\nthis is another line'
        d = self._renderAndReturnReaderResult(lambda input: input.readline(None), bytes)
        d.addCallback(self.assertEqual, b'this is one line\n')
        return d

    def test_readlineNegative(self):
        """
        Calling L{_InputStream.readline} with a negative integer as an argument
        returns one line from the input stream.
        """
        bytes = b'input stream line one\nline two'
        d = self._renderAndReturnReaderResult(lambda input: input.readline(-1), bytes)
        d.addCallback(self.assertEqual, b'input stream line one\n')
        return d

    def test_readlines(self):
        """
        Calling L{_InputStream.readlines} with no arguments returns a list of
        all lines from the input stream.
        """
        bytes = b'alice\nbob\ncarol'
        d = self._renderAndReturnReaderResult(lambda input: input.readlines(), bytes)
        d.addCallback(self.assertEqual, [b'alice\n', b'bob\n', b'carol'])
        return d

    def test_readlinesSome(self):
        """
        Calling L{_InputStream.readlines} with an integer as an argument
        returns a list of lines from the input stream with the argument serving
        as an approximate bound on the total number of bytes to read.
        """
        bytes = b'123\n456\n789\n0'
        d = self._renderAndReturnReaderResult(lambda input: input.readlines(5), bytes)

        def cbLines(lines):
            self.assertEqual(lines[:2], [b'123\n', b'456\n'])
        d.addCallback(cbLines)
        return d

    def test_readlinesMoreThan(self):
        """
        Calling L{_InputStream.readlines} with an integer which is greater than
        the total number of bytes in the input stream returns a list of all
        lines from the input.
        """
        bytes = b'one potato\ntwo potato\nthree potato'
        d = self._renderAndReturnReaderResult(lambda input: input.readlines(100), bytes)
        d.addCallback(self.assertEqual, [b'one potato\n', b'two potato\n', b'three potato'])
        return d

    def test_readlinesAfterRead(self):
        """
        Calling L{_InputStream.readlines} after a call to L{_InputStream.read}
        returns lines starting at the byte after the last byte returned by the
        C{read} call.
        """
        bytes = b'hello\nworld\nfoo'

        def readlines(input):
            input.read(7)
            return input.readlines()
        d = self._renderAndReturnReaderResult(readlines, bytes)
        d.addCallback(self.assertEqual, [b'orld\n', b'foo'])
        return d

    def test_readlinesNone(self):
        """
        Calling L{_InputStream.readlines} with L{None} as an argument returns
        all lines from the input.
        """
        bytes = b'one fish\ntwo fish\n'
        d = self._renderAndReturnReaderResult(lambda input: input.readlines(None), bytes)
        d.addCallback(self.assertEqual, [b'one fish\n', b'two fish\n'])
        return d

    def test_readlinesNegative(self):
        """
        Calling L{_InputStream.readlines} with a negative integer as an
        argument returns a list of all lines from the input.
        """
        bytes = b'red fish\nblue fish\n'
        d = self._renderAndReturnReaderResult(lambda input: input.readlines(-1), bytes)
        d.addCallback(self.assertEqual, [b'red fish\n', b'blue fish\n'])
        return d

    def test_iterable(self):
        """
        Iterating over L{_InputStream} produces lines from the input stream.
        """
        bytes = b'green eggs\nand ham\n'
        d = self._renderAndReturnReaderResult(lambda input: list(input), bytes)
        d.addCallback(self.assertEqual, [b'green eggs\n', b'and ham\n'])
        return d

    def test_iterableAfterRead(self):
        """
        Iterating over L{_InputStream} after calling L{_InputStream.read}
        produces lines from the input stream starting from the first byte after
        the last byte returned by the C{read} call.
        """
        bytes = b'green eggs\nand ham\n'

        def iterate(input):
            input.read(3)
            return list(input)
        d = self._renderAndReturnReaderResult(iterate, bytes)
        d.addCallback(self.assertEqual, [b'en eggs\n', b'and ham\n'])
        return d