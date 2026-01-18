import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
@skipIf(runtime.platform.getType() != 'win32', 'Only runs on Windows')
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class Win32ProcessTests(unittest.TestCase):
    """
    Test process programs that are packaged with twisted.
    """

    def _test_stdinReader(self, pyExe, args, env, path):
        """
        Spawn a process, write to stdin, and check the output.
        """
        p = Accumulator()
        d = p.endedDeferred = defer.Deferred()
        reactor.spawnProcess(p, pyExe, args, env, path)
        p.transport.write(b'hello, world')
        p.transport.closeStdin()

        def processEnded(ign):
            self.assertEqual(p.errF.getvalue(), b'err\nerr\n')
            self.assertEqual(p.outF.getvalue(), b'out\nhello, world\nout\n')
        return d.addCallback(processEnded)

    def test_stdinReader_bytesArgs(self):
        """
        Pass L{bytes} args to L{_test_stdinReader}.
        """
        import win32api
        pyExe = FilePath(sys.executable)._asBytesPath()
        args = [pyExe, b'-u', b'-m', b'twisted.test.process_stdinreader']
        env = dict(os.environ)
        env[b'PYTHONPATH'] = os.pathsep.join(sys.path).encode(sys.getfilesystemencoding())
        path = win32api.GetTempPath()
        path = path.encode(sys.getfilesystemencoding())
        d = self._test_stdinReader(pyExe, args, env, path)
        return d

    def test_stdinReader_unicodeArgs(self):
        """
        Pass L{unicode} args to L{_test_stdinReader}.
        """
        import win32api
        pyExe = FilePath(sys.executable).path
        args = [pyExe, '-u', '-m', 'twisted.test.process_stdinreader']
        env = properEnv
        pythonPath = os.pathsep.join(sys.path)
        env['PYTHONPATH'] = pythonPath
        path = win32api.GetTempPath()
        d = self._test_stdinReader(pyExe, args, env, path)
        return d

    def test_badArgs(self):
        pyArgs = [pyExe, b'-u', b'-c', b"print('hello')"]
        p = Accumulator()
        self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, uid=1)
        self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, gid=1)
        self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, usePTY=1)
        self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, childFDs={1: 'r'})

    def _testSignal(self, sig):
        scriptPath = b'twisted.test.process_signal'
        d = defer.Deferred()
        p = Win32SignalProtocol(d, sig)
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv)
        return d

    def test_signalTERM(self):
        """
        Sending the SIGTERM signal terminates a created process, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} attribute set to 1.
        """
        return self._testSignal('TERM')

    def test_signalINT(self):
        """
        Sending the SIGINT signal terminates a created process, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} attribute set to 1.
        """
        return self._testSignal('INT')

    def test_signalKILL(self):
        """
        Sending the SIGKILL signal terminates a created process, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} attribute set to 1.
        """
        return self._testSignal('KILL')

    def test_closeHandles(self):
        """
        The win32 handles should be properly closed when the process exits.
        """
        import win32api
        connected = defer.Deferred()
        ended = defer.Deferred()

        class SimpleProtocol(protocol.ProcessProtocol):
            """
            A protocol that fires deferreds when connected and disconnected.
            """

            def makeConnection(self, transport):
                connected.callback(transport)

            def processEnded(self, reason):
                ended.callback(None)
        p = SimpleProtocol()
        pyArgs = [pyExe, b'-u', b'-c', b"print('hello')"]
        proc = reactor.spawnProcess(p, pyExe, pyArgs)

        def cbConnected(transport):
            self.assertIs(transport, proc)
            win32api.GetHandleInformation(proc.hProcess)
            win32api.GetHandleInformation(proc.hThread)
            self.hProcess = proc.hProcess
            self.hThread = proc.hThread
        connected.addCallback(cbConnected)

        def checkTerminated(ignored):
            self.assertIsNone(proc.pid)
            self.assertIsNone(proc.hProcess)
            self.assertIsNone(proc.hThread)
            self.assertRaises(win32api.error, win32api.GetHandleInformation, self.hProcess)
            self.assertRaises(win32api.error, win32api.GetHandleInformation, self.hThread)
        ended.addCallback(checkTerminated)
        return defer.gatherResults([connected, ended])