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
class PosixProcessBase:
    """
    Test running processes.
    """
    usePTY = False

    def getCommand(self, commandName):
        """
        Return the path of the shell command named C{commandName}, looking at
        common locations.
        """
        for loc in procutils.which(commandName):
            return FilePath(loc).asBytesMode().path
        binLoc = FilePath('/bin').child(commandName)
        usrbinLoc = FilePath('/usr/bin').child(commandName)
        if binLoc.exists():
            return binLoc.asBytesMode().path
        elif usrbinLoc.exists():
            return usrbinLoc.asBytesMode().path
        else:
            raise RuntimeError(f'{commandName} found in neither standard location nor on PATH ({os.environ['PATH']})')

    def test_normalTermination(self):
        cmd = self.getCommand('true')
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        reactor.spawnProcess(p, cmd, [b'true'], env=None, usePTY=self.usePTY)

        def check(ignored):
            p.reason.trap(error.ProcessDone)
            self.assertEqual(p.reason.value.exitCode, 0)
            self.assertIsNone(p.reason.value.signal)
        d.addCallback(check)
        return d

    def test_abnormalTermination(self):
        """
        When a process terminates with a system exit code set to 1,
        C{processEnded} is called with a L{error.ProcessTerminated} error,
        the C{exitCode} attribute reflecting the system exit code.
        """
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        reactor.spawnProcess(p, pyExe, [pyExe, b'-c', b'import sys; sys.exit(1)'], env=None, usePTY=self.usePTY)

        def check(ignored):
            p.reason.trap(error.ProcessTerminated)
            self.assertEqual(p.reason.value.exitCode, 1)
            self.assertIsNone(p.reason.value.signal)
        d.addCallback(check)
        return d

    def _testSignal(self, sig):
        scriptPath = b'twisted.test.process_signal'
        d = defer.Deferred()
        p = SignalProtocol(d, sig)
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', '-m', scriptPath], env=properEnv, usePTY=self.usePTY)
        return d

    def test_signalHUP(self):
        """
        Sending the SIGHUP signal to a running process interrupts it, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} set to L{None} and the C{signal} attribute set to
        C{signal.SIGHUP}. C{os.WTERMSIG} can also be used on the C{status}
        attribute to extract the signal value.
        """
        return self._testSignal('HUP')

    def test_signalINT(self):
        """
        Sending the SIGINT signal to a running process interrupts it, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} set to L{None} and the C{signal} attribute set to
        C{signal.SIGINT}. C{os.WTERMSIG} can also be used on the C{status}
        attribute to extract the signal value.
        """
        return self._testSignal('INT')

    def test_signalKILL(self):
        """
        Sending the SIGKILL signal to a running process interrupts it, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} set to L{None} and the C{signal} attribute set to
        C{signal.SIGKILL}. C{os.WTERMSIG} can also be used on the C{status}
        attribute to extract the signal value.
        """
        return self._testSignal('KILL')

    def test_signalTERM(self):
        """
        Sending the SIGTERM signal to a running process interrupts it, and
        C{processEnded} is called with a L{error.ProcessTerminated} instance
        with the C{exitCode} set to L{None} and the C{signal} attribute set to
        C{signal.SIGTERM}. C{os.WTERMSIG} can also be used on the C{status}
        attribute to extract the signal value.
        """
        return self._testSignal('TERM')

    def test_childSignalHandling(self):
        """
        The disposition of signals which are ignored in the parent
        process is reset to the default behavior for the child
        process.
        """
        which = signal.SIGUSR1
        handler = signal.signal(which, signal.SIG_IGN)
        self.addCleanup(signal.signal, signal.SIGUSR1, handler)
        return self._testSignal(signal.SIGUSR1)

    def test_executionError(self):
        """
        Raise an error during execvpe to check error management.
        """
        if runtime.platform.isMacOSX() and self.usePTY:
            raise SkipTest('Test is flaky from a Darwin bug. See #8840.')
        cmd = self.getCommand('false')
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)

        def buggyexecvpe(command, args, environment):
            raise RuntimeError('Ouch')
        oldexecvpe = os.execvpe
        os.execvpe = buggyexecvpe
        reactor._neverUseSpawn = True
        try:
            reactor.spawnProcess(p, cmd, [b'false'], env=None, usePTY=self.usePTY)

            def check(ignored):
                errData = b''.join(p.errData + p.outData)
                self.assertIn(b'Upon execvpe', errData)
                self.assertIn(b'Ouch', errData)
            d.addCallback(check)
        finally:
            os.execvpe = oldexecvpe
        return d

    def test_errorInProcessEnded(self):
        """
        The handler which reaps a process is removed when the process is
        reaped, even if the protocol's C{processEnded} method raises an
        exception.
        """
        connected = defer.Deferred()
        ended = defer.Deferred()
        scriptPath = b'twisted.test.process_echoer'

        class ErrorInProcessEnded(protocol.ProcessProtocol):
            """
            A protocol that raises an error in C{processEnded}.
            """

            def makeConnection(self, transport):
                connected.callback(transport)

            def processEnded(self, reason):
                reactor.callLater(0, ended.callback, None)
                raise RuntimeError('Deliberate error')
        reactor.spawnProcess(ErrorInProcessEnded(), pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, path=None)
        pid = []

        def cbConnected(transport):
            pid.append(transport.pid)
            self.assertIn(transport.pid, process.reapProcessHandlers)
            transport.loseConnection()
        connected.addCallback(cbConnected)

        def checkTerminated(ignored):
            excs = self.flushLoggedErrors(RuntimeError)
            self.assertEqual(len(excs), 1)
            self.assertNotIn(pid[0], process.reapProcessHandlers)
        ended.addCallback(checkTerminated)
        return ended