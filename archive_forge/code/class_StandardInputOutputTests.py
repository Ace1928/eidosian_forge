import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
class StandardInputOutputTests(TestCase):
    if platform.isWindows() and requireModule('win32process') is None:
        skip = 'On windows, spawnProcess is not available in the absence of win32process.'

    def _spawnProcess(self, proto, sibling, *args, **kw):
        """
        Launch a child Python process and communicate with it using the
        given ProcessProtocol.

        @param proto: A L{ProcessProtocol} instance which will be connected
        to the child process.

        @param sibling: The basename of a file containing the Python program
        to run in the child process.

        @param *args: strings which will be passed to the child process on
        the command line as C{argv[2:]}.

        @param **kw: additional arguments to pass to L{reactor.spawnProcess}.

        @return: The L{IProcessTransport} provider for the spawned process.
        """
        args = [sys.executable, b'-m', b'twisted.test.' + sibling, reactor.__class__.__module__] + list(args)
        return reactor.spawnProcess(proto, sys.executable, args, env=properEnv, **kw)

    def _requireFailure(self, d, callback):

        def cb(result):
            self.fail(f'Process terminated with non-Failure: {result!r}')

        def eb(err):
            return callback(err)
        return d.addCallbacks(cb, eb)

    def test_loseConnection(self):
        """
        Verify that a protocol connected to L{StandardIO} can disconnect
        itself using C{transport.loseConnection}.
        """
        errorLogFile = self.mktemp()
        log.msg('Child process logging to ' + errorLogFile)
        p = StandardIOTestProcessProtocol()
        d = p.onCompletion
        self._spawnProcess(p, b'stdio_test_loseconn', errorLogFile)

        def processEnded(reason):
            with open(errorLogFile) as f:
                for line in f:
                    log.msg('Child logged: ' + line.rstrip())
            self.failIfIn(1, p.data)
            reason.trap(error.ProcessDone)
        return self._requireFailure(d, processEnded)

    def test_readConnectionLost(self):
        """
        When stdin is closed and the protocol connected to it implements
        L{IHalfCloseableProtocol}, the protocol's C{readConnectionLost} method
        is called.
        """
        errorLogFile = self.mktemp()
        log.msg('Child process logging to ' + errorLogFile)
        p = StandardIOTestProcessProtocol()
        p.onDataReceived = defer.Deferred()

        def cbBytes(ignored):
            d = p.onCompletion
            p.transport.closeStdin()
            return d
        p.onDataReceived.addCallback(cbBytes)

        def processEnded(reason):
            reason.trap(error.ProcessDone)
        d = self._requireFailure(p.onDataReceived, processEnded)
        self._spawnProcess(p, b'stdio_test_halfclose', errorLogFile)
        return d

    def test_lastWriteReceived(self):
        """
        Verify that a write made directly to stdout using L{os.write}
        after StandardIO has finished is reliably received by the
        process reading that stdout.
        """
        p = StandardIOTestProcessProtocol()
        try:
            self._spawnProcess(p, b'stdio_test_lastwrite', UNIQUE_LAST_WRITE_STRING, usePTY=True)
        except ValueError as e:
            raise SkipTest(str(e))

        def processEnded(reason):
            """
            Asserts that the parent received the bytes written by the child
            immediately after the child starts.
            """
            self.assertTrue(p.data[1].endswith(UNIQUE_LAST_WRITE_STRING), f'Received {p.data!r} from child, did not find expected bytes.')
            reason.trap(error.ProcessDone)
        return self._requireFailure(p.onCompletion, processEnded)

    def test_hostAndPeer(self):
        """
        Verify that the transport of a protocol connected to L{StandardIO}
        has C{getHost} and C{getPeer} methods.
        """
        p = StandardIOTestProcessProtocol()
        d = p.onCompletion
        self._spawnProcess(p, b'stdio_test_hostpeer')

        def processEnded(reason):
            host, peer = p.data[1].splitlines()
            self.assertTrue(host)
            self.assertTrue(peer)
            reason.trap(error.ProcessDone)
        return self._requireFailure(d, processEnded)

    def test_write(self):
        """
        Verify that the C{write} method of the transport of a protocol
        connected to L{StandardIO} sends bytes to standard out.
        """
        p = StandardIOTestProcessProtocol()
        d = p.onCompletion
        self._spawnProcess(p, b'stdio_test_write')

        def processEnded(reason):
            self.assertEqual(p.data[1], b'ok!')
            reason.trap(error.ProcessDone)
        return self._requireFailure(d, processEnded)

    def test_writeSequence(self):
        """
        Verify that the C{writeSequence} method of the transport of a
        protocol connected to L{StandardIO} sends bytes to standard out.
        """
        p = StandardIOTestProcessProtocol()
        d = p.onCompletion
        self._spawnProcess(p, b'stdio_test_writeseq')

        def processEnded(reason):
            self.assertEqual(p.data[1], b'ok!')
            reason.trap(error.ProcessDone)
        return self._requireFailure(d, processEnded)

    def _junkPath(self):
        junkPath = self.mktemp()
        with open(junkPath, 'wb') as junkFile:
            for i in range(1024):
                junkFile.write(b'%d\n' % (i,))
        return junkPath

    def test_producer(self):
        """
        Verify that the transport of a protocol connected to L{StandardIO}
        is a working L{IProducer} provider.
        """
        p = StandardIOTestProcessProtocol()
        d = p.onCompletion
        written = []
        toWrite = list(range(100))

        def connectionMade(ign):
            if toWrite:
                written.append(b'%d\n' % (toWrite.pop(),))
                proc.write(written[-1])
                reactor.callLater(0.01, connectionMade, None)
        proc = self._spawnProcess(p, b'stdio_test_producer')
        p.onConnection.addCallback(connectionMade)

        def processEnded(reason):
            self.assertEqual(p.data[1], b''.join(written))
            self.assertFalse(toWrite, 'Connection lost with %d writes left to go.' % (len(toWrite),))
            reason.trap(error.ProcessDone)
        return self._requireFailure(d, processEnded)

    def test_consumer(self):
        """
        Verify that the transport of a protocol connected to L{StandardIO}
        is a working L{IConsumer} provider.
        """
        p = StandardIOTestProcessProtocol()
        d = p.onCompletion
        junkPath = self._junkPath()
        self._spawnProcess(p, b'stdio_test_consumer', junkPath)

        def processEnded(reason):
            with open(junkPath, 'rb') as f:
                self.assertEqual(p.data[1], f.read())
            reason.trap(error.ProcessDone)
        return self._requireFailure(d, processEnded)

    @skipIf(platform.isWindows(), 'StandardIO does not accept stdout as an argument to Windows.  Testing redirection to a file is therefore harder.')
    def test_normalFileStandardOut(self):
        """
        If L{StandardIO} is created with a file descriptor which refers to a
        normal file (ie, a file from the filesystem), L{StandardIO.write}
        writes bytes to that file.  In particular, it does not immediately
        consider the file closed or call its protocol's C{connectionLost}
        method.
        """
        onConnLost = defer.Deferred()
        proto = ConnectionLostNotifyingProtocol(onConnLost)
        path = filepath.FilePath(self.mktemp())
        self.normal = normal = path.open('wb')
        self.addCleanup(normal.close)
        kwargs = dict(stdout=normal.fileno())
        if not platform.isWindows():
            r, w = os.pipe()
            self.addCleanup(os.close, r)
            self.addCleanup(os.close, w)
            kwargs['stdin'] = r
        connection = stdio.StandardIO(proto, **kwargs)
        howMany = 5
        count = itertools.count()

        def spin():
            for value in count:
                if value == howMany:
                    connection.loseConnection()
                    return
                connection.write(b'%d' % (value,))
                break
            reactor.callLater(0, spin)
        reactor.callLater(0, spin)

        def cbLost(reason):
            self.assertEqual(next(count), howMany + 1)
            self.assertEqual(path.getContent(), b''.join((b'%d' % (i,) for i in range(howMany))))
        onConnLost.addCallback(cbLost)
        return onConnLost