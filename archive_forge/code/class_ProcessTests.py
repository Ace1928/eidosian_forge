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
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class ProcessTests(unittest.TestCase):
    """
    Test running a process.
    """
    usePTY = False

    def test_stdio(self):
        """
        L{twisted.internet.stdio} test.
        """
        scriptPath = 'twisted.test.process_twisted'
        p = Accumulator()
        d = p.endedDeferred = defer.Deferred()
        reactor.spawnProcess(p, pyExe, [pyExe, '-u', '-m', scriptPath], env=properEnv, path=None, usePTY=self.usePTY)
        p.transport.write(b'hello, world')
        p.transport.write(b'abc')
        p.transport.write(b'123')
        p.transport.closeStdin()

        def processEnded(ign):
            self.assertEqual(p.outF.getvalue(), b'hello, worldabc123', 'Output follows:\n%s\nError message from process_twisted follows:\n%s\n' % (p.outF.getvalue(), p.errF.getvalue()))
        return d.addCallback(processEnded)

    def test_patchSysStdoutWithNone(self):
        """
        In some scenarious, such as Python running as part of a Windows
        Windows GUI Application with no console, L{sys.stdout} is L{None}.
        """
        import sys
        self.patch(sys, 'stdout', None)
        return self.test_stdio()

    def test_patchSysStdoutWithStringIO(self):
        """
        Some projects which use the Twisted reactor
        such as Buildbot patch L{sys.stdout} with L{io.StringIO}
        before running their tests.
        """
        import sys
        from io import StringIO
        stdoutStringIO = StringIO()
        self.patch(sys, 'stdout', stdoutStringIO)
        return self.test_stdio()

    def test_patch_sys__stdout__WithStringIO(self):
        """
        If L{sys.stdout} and L{sys.__stdout__} are patched with L{io.StringIO},
        we should get a L{ValueError}.
        """
        import sys
        from io import StringIO
        self.patch(sys, 'stdout', StringIO())
        self.patch(sys, '__stdout__', StringIO())
        return self.test_stdio()

    def test_unsetPid(self):
        """
        Test if pid is None/non-None before/after process termination.  This
        reuses process_echoer.py to get a process that blocks on stdin.
        """
        finished = defer.Deferred()
        p = TrivialProcessProtocol(finished)
        scriptPath = b'twisted.test.process_echoer'
        procTrans = reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv)
        self.assertTrue(procTrans.pid)

        def afterProcessEnd(ignored):
            self.assertIsNone(procTrans.pid)
        p.transport.closeStdin()
        return finished.addCallback(afterProcessEnd)

    @skipIf(os.environ.get('CI', '').lower() == 'true' and runtime.platform.getType() == 'win32', 'See https://twistedmatrix.com/trac/ticket/10014')
    def test_process(self):
        """
        Test running a process: check its output, it exitCode, some property of
        signalProcess.
        """
        scriptPath = b'twisted.test.process_tester'
        d = defer.Deferred()
        p = TestProcessProtocol()
        p.deferred = d
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv)

        def check(ignored):
            self.assertEqual(p.stages, [1, 2, 3, 4, 5])
            f = p.reason
            f.trap(error.ProcessTerminated)
            self.assertEqual(f.value.exitCode, 23)
            self.assertRaises(error.ProcessExitedAlready, p.transport.signalProcess, 'INT')
            try:
                import glob
                import process_tester
                for f in glob.glob(process_tester.test_file_match):
                    os.remove(f)
            except BaseException:
                pass
        d.addCallback(check)
        return d

    @skipIf(os.environ.get('CI', '').lower() == 'true' and runtime.platform.getType() == 'win32', 'See https://twistedmatrix.com/trac/ticket/10014')
    def test_manyProcesses(self):

        def _check(results, protocols):
            for p in protocols:
                self.assertEqual(p.stages, [1, 2, 3, 4, 5], '[%d] stages = %s' % (id(p.transport), str(p.stages)))
                f = p.reason
                f.trap(error.ProcessTerminated)
                self.assertEqual(f.value.exitCode, 23)
        scriptPath = b'twisted.test.process_tester'
        args = [pyExe, b'-u', b'-m', scriptPath]
        protocols = []
        deferreds = []
        for i in range(CONCURRENT_PROCESS_TEST_COUNT):
            p = TestManyProcessProtocol()
            protocols.append(p)
            reactor.spawnProcess(p, pyExe, args, env=properEnv)
            deferreds.append(p.deferred)
        deferredList = defer.DeferredList(deferreds, consumeErrors=True)
        deferredList.addCallback(_check, protocols)
        return deferredList

    def test_echo(self):
        """
        A spawning a subprocess which echoes its stdin to its stdout via
        L{IReactorProcess.spawnProcess} will result in that echoed output being
        delivered to outReceived.
        """
        finished = defer.Deferred()
        p = EchoProtocol(finished)
        scriptPath = b'twisted.test.process_echoer'
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv)

        def asserts(ignored):
            self.assertFalse(p.failure, p.failure)
            self.assertTrue(hasattr(p, 'buffer'))
            self.assertEqual(len(p.buffer), len(p.s * p.n))

        def takedownProcess(err):
            p.transport.closeStdin()
            return err
        return finished.addCallback(asserts).addErrback(takedownProcess)

    def test_commandLine(self):
        args = [b'a\\"b ', b'a\\b ', b' a\\\\"b', b' a\\\\b', b'"foo bar" "', b'\tab', b'"\\', b'a"b', b"a'b"]
        scriptPath = b'twisted.test.process_cmdline'
        p = Accumulator()
        d = p.endedDeferred = defer.Deferred()
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath] + args, env=properEnv, path=None)

        def processEnded(ign):
            self.assertEqual(p.errF.getvalue(), b'')
            recvdArgs = p.outF.getvalue().splitlines()
            self.assertEqual(recvdArgs, args)
        return d.addCallback(processEnded)