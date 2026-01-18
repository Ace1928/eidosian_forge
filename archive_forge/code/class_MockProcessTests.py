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
class MockProcessTests(unittest.TestCase):
    """
    Mock a process runner to test forked child code path.
    """
    if process is None:
        skip = 'twisted.internet.process is never used on Windows'

    def setUp(self):
        """
        Replace L{process} os, fcntl, sys, switchUID, fdesc and pty modules
        with the mock class L{MockOS}.
        """
        if gc.isenabled():
            self.addCleanup(gc.enable)
        else:
            self.addCleanup(gc.disable)
        self.mockos = MockOS()
        self.mockos.euid = 1236
        self.mockos.egid = 1234
        self.patch(process, 'os', self.mockos)
        self.patch(process, 'fcntl', self.mockos)
        self.patch(process, 'sys', self.mockos)
        self.patch(process, 'switchUID', self.mockos.switchUID)
        self.patch(process, 'fdesc', self.mockos)
        self.patch(process.Process, 'processReaderFactory', DumbProcessReader)
        self.patch(process.Process, 'processWriterFactory', DumbProcessWriter)
        self.patch(process.Process, '_trySpawnInsteadOfFork', lambda *a, **k: False)
        self.patch(process, 'pty', self.mockos)
        self.mocksig = MockSignal()
        self.patch(process, 'signal', self.mocksig)

    def tearDown(self):
        """
        Reset processes registered for reap.
        """
        process.reapProcessHandlers = {}

    def assertProcessLaunched(self):
        """
        A process should have been launched, but I don't care whether it was
        with fork() or posix_spawnp().
        """
        self.assertEqual(self.mockos.actions, [ForkOrSpawn(), 'waitpid'])

    def test_mockFork(self):
        """
        Test a classic spawnProcess. Check the path of the client code:
        fork, exec, exit.
        """
        gc.enable()
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        try:
            reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        except SystemError:
            self.assertTrue(self.mockos.exited)
            self.assertEqual(self.mockos.actions, [('fork', False), 'exec', ('exit', 1)])
        else:
            self.fail('Should not be here')
        self.assertFalse(gc.isenabled())

    def _mockForkInParentTest(self):
        """
        Assert that in the main process, spawnProcess disables the garbage
        collector, calls fork, closes the pipe file descriptors it created for
        the child process, and calls waitpid.
        """
        self.mockos.child = False
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        self.assertEqual(set(self.mockos.closed), {-1, -4, -6})
        self.assertProcessLaunched()

    def test_mockForkInParentGarbageCollectorEnabled(self):
        """
        The garbage collector should be enabled when L{reactor.spawnProcess}
        returns if it was initially enabled.

        @see L{_mockForkInParentTest}
        """
        gc.enable()
        self._mockForkInParentTest()
        self.assertTrue(gc.isenabled())

    def test_mockForkInParentGarbageCollectorDisabled(self):
        """
        The garbage collector should be disabled when L{reactor.spawnProcess}
        returns if it was initially disabled.

        @see L{_mockForkInParentTest}
        """
        gc.disable()
        self._mockForkInParentTest()
        self.assertFalse(gc.isenabled())

    def test_mockForkTTY(self):
        """
        Test a TTY spawnProcess: check the path of the client code:
        fork, exec, exit.
        """
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        self.assertRaises(SystemError, reactor.spawnProcess, p, cmd, [b'ouch'], env=None, usePTY=True)
        self.assertTrue(self.mockos.exited)
        self.assertEqual(self.mockos.actions, [('fork', False), 'setsid', 'exec', ('exit', 1)])

    def _mockWithForkError(self):
        """
        Assert that if the fork call fails, no other process setup calls are
        made and that spawnProcess raises the exception fork raised.
        """
        self.mockos.raiseFork = OSError(errno.EAGAIN, None)
        protocol = TrivialProcessProtocol(None)
        self.assertRaises(OSError, reactor.spawnProcess, protocol, None)
        self.assertEqual(self.mockos.actions, [('fork', False)])

    def test_mockWithForkErrorGarbageCollectorEnabled(self):
        """
        The garbage collector should be enabled when L{reactor.spawnProcess}
        raises because L{os.fork} raised, if it was initially enabled.
        """
        gc.enable()
        self._mockWithForkError()
        self.assertTrue(gc.isenabled())

    def test_mockWithForkErrorGarbageCollectorDisabled(self):
        """
        The garbage collector should be disabled when
        L{reactor.spawnProcess} raises because L{os.fork} raised, if it was
        initially disabled.
        """
        gc.disable()
        self._mockWithForkError()
        self.assertFalse(gc.isenabled())

    def test_mockForkErrorCloseFDs(self):
        """
        When C{os.fork} raises an exception, the file descriptors created
        before are closed and don't leak.
        """
        self._mockWithForkError()
        self.assertEqual(set(self.mockos.closed), {-1, -4, -6, -2, -3, -5})

    def test_mockForkErrorGivenFDs(self):
        """
        When C{os.forks} raises an exception and that file descriptors have
        been specified with the C{childFDs} arguments of
        L{reactor.spawnProcess}, they are not closed.
        """
        self.mockos.raiseFork = OSError(errno.EAGAIN, None)
        protocol = TrivialProcessProtocol(None)
        self.assertRaises(OSError, reactor.spawnProcess, protocol, None, childFDs={0: -10, 1: -11, 2: -13})
        self.assertEqual(self.mockos.actions, [('fork', False)])
        self.assertEqual(self.mockos.closed, [])
        self.assertRaises(OSError, reactor.spawnProcess, protocol, None, childFDs={0: 'r', 1: -11, 2: -13})
        self.assertEqual(set(self.mockos.closed), {-1, -2})

    def test_mockForkErrorClosePTY(self):
        """
        When C{os.fork} raises an exception, the file descriptors created by
        C{pty.openpty} are closed and don't leak, when C{usePTY} is set to
        C{True}.
        """
        self.mockos.raiseFork = OSError(errno.EAGAIN, None)
        protocol = TrivialProcessProtocol(None)
        self.assertRaises(OSError, reactor.spawnProcess, protocol, None, usePTY=True)
        self.assertEqual(self.mockos.actions, [('fork', False)])
        self.assertEqual(set(self.mockos.closed), {-12, -13})

    def test_mockForkErrorPTYGivenFDs(self):
        """
        If a tuple is passed to C{usePTY} to specify slave and master file
        descriptors and that C{os.fork} raises an exception, these file
        descriptors aren't closed.
        """
        self.mockos.raiseFork = OSError(errno.EAGAIN, None)
        protocol = TrivialProcessProtocol(None)
        self.assertRaises(OSError, reactor.spawnProcess, protocol, None, usePTY=(-20, -21, 'foo'))
        self.assertEqual(self.mockos.actions, [('fork', False)])
        self.assertEqual(self.mockos.closed, [])

    def test_mockWithExecError(self):
        """
        Spawn a process but simulate an error during execution in the client
        path: C{os.execvpe} raises an error. It should close all the standard
        fds, try to print the error encountered, and exit cleanly.
        """
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        self.mockos.raiseExec = True
        try:
            reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        except SystemError:
            self.assertTrue(self.mockos.exited)
            self.assertEqual(self.mockos.actions, [('fork', False), 'exec', ('exit', 1)])
            self.assertIn(0, self.mockos.closed)
            self.assertIn(1, self.mockos.closed)
            self.assertIn(2, self.mockos.closed)
            self.assertIn(b'RuntimeError: Bar', self.mockos.fdio.getvalue())
        else:
            self.fail('Should not be here')

    def test_mockSetUid(self):
        """
        Try creating a process with setting its uid: it's almost the same path
        as the standard path, but with a C{switchUID} call before the exec.
        """
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        try:
            reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False, uid=8080)
        except SystemError:
            self.assertTrue(self.mockos.exited)
            self.assertEqual(self.mockos.actions, [('fork', False), ('setuid', 0), ('setgid', 0), ('switchuid', 8080, 1234), 'exec', ('exit', 1)])
        else:
            self.fail('Should not be here')

    def test_mockSetUidInParent(self):
        """
        When spawning a child process with a UID different from the UID of the
        current process, the current process does not have its UID changed.
        """
        self.mockos.child = False
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False, uid=8080)
        self.assertProcessLaunched()

    def test_mockPTYSetUid(self):
        """
        Try creating a PTY process with setting its uid: it's almost the same
        path as the standard path, but with a C{switchUID} call before the
        exec.
        """
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        try:
            reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=True, uid=8081)
        except SystemError:
            self.assertTrue(self.mockos.exited)
            self.assertEqual(self.mockos.actions, [('fork', False), 'setsid', ('setuid', 0), ('setgid', 0), ('switchuid', 8081, 1234), 'exec', ('exit', 1)])
        else:
            self.fail('Should not be here')

    def test_mockPTYSetUidInParent(self):
        """
        When spawning a child process with PTY and a UID different from the UID
        of the current process, the current process does not have its UID
        changed.
        """
        self.mockos.child = False
        cmd = b'/mock/ouch'
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        oldPTYProcess = process.PTYProcess
        try:
            process.PTYProcess = DumbPTYProcess
            reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=True, uid=8080)
        finally:
            process.PTYProcess = oldPTYProcess
        self.assertProcessLaunched()

    def test_mockWithWaitError(self):
        """
        Test that reapProcess logs errors raised.
        """
        self.mockos.child = False
        cmd = b'/mock/ouch'
        self.mockos.waitChild = (0, 0)
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        self.assertProcessLaunched()
        self.mockos.raiseWaitPid = OSError()
        proc.reapProcess()
        errors = self.flushLoggedErrors()
        self.assertEqual(len(errors), 1)
        errors[0].trap(OSError)

    def test_mockErrorECHILDInReapProcess(self):
        """
        Test that reapProcess doesn't log anything when waitpid raises a
        C{OSError} with errno C{ECHILD}.
        """
        self.mockos.child = False
        cmd = b'/mock/ouch'
        self.mockos.waitChild = (0, 0)
        d = defer.Deferred()
        p = TrivialProcessProtocol(d)
        proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        self.assertProcessLaunched()
        self.mockos.raiseWaitPid = OSError()
        self.mockos.raiseWaitPid.errno = errno.ECHILD
        proc.reapProcess()

    def test_mockErrorInPipe(self):
        """
        If C{os.pipe} raises an exception after some pipes where created, the
        created pipes are closed and don't leak.
        """
        pipes = [-1, -2, -3, -4]

        def pipe():
            try:
                return (pipes.pop(0), pipes.pop(0))
            except IndexError:
                raise OSError()
        self.mockos.pipe = pipe
        protocol = TrivialProcessProtocol(None)
        self.assertRaises(OSError, reactor.spawnProcess, protocol, None)
        self.assertEqual(self.mockos.actions, [])
        self.assertEqual(set(self.mockos.closed), {-4, -3, -2, -1})

    def test_kill(self):
        """
        L{process.Process.signalProcess} calls C{os.kill} translating the given
        signal string to the PID.
        """
        self.mockos.child = False
        self.mockos.waitChild = (0, 0)
        cmd = b'/mock/ouch'
        p = TrivialProcessProtocol(None)
        proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        proc.signalProcess('KILL')
        self.assertEqual(self.mockos.actions, [('fork', False), 'waitpid', ('kill', 21, signal.SIGKILL)])

    def test_killExited(self):
        """
        L{process.Process.signalProcess} raises L{error.ProcessExitedAlready}
        if the process has exited.
        """
        self.mockos.child = False
        cmd = b'/mock/ouch'
        p = TrivialProcessProtocol(None)
        proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        self.assertRaises(error.ProcessExitedAlready, proc.signalProcess, 'KILL')

    def test_killExitedButNotDetected(self):
        """
        L{process.Process.signalProcess} raises L{error.ProcessExitedAlready}
        if the process has exited but that twisted hasn't seen it (for example,
        if the process has been waited outside of twisted): C{os.kill} then
        raise C{OSError} with C{errno.ESRCH} as errno.
        """
        self.mockos.child = False
        self.mockos.waitChild = (0, 0)
        cmd = b'/mock/ouch'
        p = TrivialProcessProtocol(None)
        proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        self.mockos.raiseKill = OSError(errno.ESRCH, 'Not found')
        self.assertRaises(error.ProcessExitedAlready, proc.signalProcess, 'KILL')

    def test_killErrorInKill(self):
        """
        L{process.Process.signalProcess} doesn't mask C{OSError} exceptions if
        the errno is different from C{errno.ESRCH}.
        """
        self.mockos.child = False
        self.mockos.waitChild = (0, 0)
        cmd = b'/mock/ouch'
        p = TrivialProcessProtocol(None)
        proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
        self.mockos.raiseKill = OSError(errno.EINVAL, 'Invalid signal')
        err = self.assertRaises(OSError, proc.signalProcess, 'KILL')
        self.assertEqual(err.errno, errno.EINVAL)