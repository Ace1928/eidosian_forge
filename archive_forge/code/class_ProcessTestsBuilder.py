import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
class ProcessTestsBuilder(ProcessTestsBuilderBase):
    """
    Builder defining tests relating to L{IReactorProcess} for child processes
    which do not have a PTY.
    """
    usePTY = False
    keepStdioOpenProgram = b'twisted.internet.test.process_helper'
    if platform.isWindows():
        keepStdioOpenArg = b'windows'
    else:
        keepStdioOpenArg = b''

    def test_childConnectionLost(self):
        """
        L{IProcessProtocol.childConnectionLost} is called each time a file
        descriptor associated with a child process is closed.
        """
        connected = Deferred()
        lost = {0: Deferred(), 1: Deferred(), 2: Deferred()}

        class Closer(ProcessProtocol):

            def makeConnection(self, transport):
                connected.callback(transport)

            def childConnectionLost(self, childFD):
                lost[childFD].callback(None)
        target = b'twisted.internet.test.process_loseconnection'
        reactor = self.buildReactor()
        reactor.callWhenRunning(reactor.spawnProcess, Closer(), pyExe, [pyExe, b'-m', target], env=properEnv, usePTY=self.usePTY)

        def cbConnected(transport):
            transport.write(b'2\n')
            return lost[2].addCallback(lambda ign: transport)
        connected.addCallback(cbConnected)

        def lostSecond(transport):
            transport.write(b'1\n')
            return lost[1].addCallback(lambda ign: transport)
        connected.addCallback(lostSecond)

        def lostFirst(transport):
            transport.write(b'\n')
        connected.addCallback(lostFirst)
        connected.addErrback(err)

        def cbEnded(ignored):
            reactor.stop()
        connected.addCallback(cbEnded)
        self.runReactor(reactor)

    def test_processEnded(self):
        """
        L{IProcessProtocol.processEnded} is called after the child process
        exits and L{IProcessProtocol.childConnectionLost} is called for each of
        its file descriptors.
        """
        ended = Deferred()
        lost = []

        class Ender(ProcessProtocol):

            def childDataReceived(self, fd, data):
                msg('childDataReceived(%d, %r)' % (fd, data))
                self.transport.loseConnection()

            def childConnectionLost(self, childFD):
                msg('childConnectionLost(%d)' % (childFD,))
                lost.append(childFD)

            def processExited(self, reason):
                msg(f'processExited({reason!r})')

            def processEnded(self, reason):
                msg(f'processEnded({reason!r})')
                ended.callback([reason])
        reactor = self.buildReactor()
        reactor.callWhenRunning(reactor.spawnProcess, Ender(), pyExe, [pyExe, b'-m', self.keepStdioOpenProgram, b'child', self.keepStdioOpenArg], env=properEnv, usePTY=self.usePTY)

        def cbEnded(args):
            failure, = args
            failure.trap(ProcessDone)
            self.assertEqual(set(lost), {0, 1, 2})
        ended.addCallback(cbEnded)
        ended.addErrback(err)
        ended.addCallback(lambda ign: reactor.stop())
        self.runReactor(reactor)

    def test_processExited(self):
        """
        L{IProcessProtocol.processExited} is called when the child process
        exits, even if file descriptors associated with the child are still
        open.
        """
        exited = Deferred()
        allLost = Deferred()
        lost = []

        class Waiter(ProcessProtocol):

            def childDataReceived(self, fd, data):
                msg('childDataReceived(%d, %r)' % (fd, data))

            def childConnectionLost(self, childFD):
                msg('childConnectionLost(%d)' % (childFD,))
                lost.append(childFD)
                if len(lost) == 3:
                    allLost.callback(None)

            def processExited(self, reason):
                msg(f'processExited({reason!r})')
                exited.callback([reason])
                self.transport.loseConnection()
        reactor = self.buildReactor()
        reactor.callWhenRunning(reactor.spawnProcess, Waiter(), pyExe, [pyExe, b'-u', b'-m', self.keepStdioOpenProgram, b'child', self.keepStdioOpenArg], env=properEnv, usePTY=self.usePTY)

        def cbExited(args):
            failure, = args
            failure.trap(ProcessDone)
            msg(f'cbExited; lost = {lost}')
            self.assertEqual(lost, [])
            return allLost
        exited.addCallback(cbExited)

        def cbAllLost(ignored):
            self.assertEqual(set(lost), {0, 1, 2})
        exited.addCallback(cbAllLost)
        exited.addErrback(err)
        exited.addCallback(lambda ign: reactor.stop())
        self.runReactor(reactor)

    def makeSourceFile(self, sourceLines):
        """
        Write the given list of lines to a text file and return the absolute
        path to it.
        """
        script = _asFilesystemBytes(self.mktemp())
        with open(script, 'wt') as scriptFile:
            scriptFile.write(os.linesep.join(sourceLines) + os.linesep)
        return os.path.abspath(script)

    def test_shebang(self):
        """
        Spawning a process with an executable which is a script starting
        with an interpreter definition line (#!) uses that interpreter to
        evaluate the script.
        """
        shebangOutput = b'this is the shebang output'
        scriptFile = self.makeSourceFile(['#!{}'.format(pyExe.decode('ascii')), 'import sys', "sys.stdout.write('{}')".format(shebangOutput.decode('ascii')), 'sys.stdout.flush()'])
        os.chmod(scriptFile, 448)
        reactor = self.buildReactor()

        def cbProcessExited(args):
            out, err, code = args
            msg('cbProcessExited((%r, %r, %d))' % (out, err, code))
            self.assertEqual(out, shebangOutput)
            self.assertEqual(err, b'')
            self.assertEqual(code, 0)

        def shutdown(passthrough):
            reactor.stop()
            return passthrough

        def start():
            d = utils.getProcessOutputAndValue(scriptFile, reactor=reactor)
            d.addBoth(shutdown)
            d.addCallback(cbProcessExited)
            d.addErrback(err)
        reactor.callWhenRunning(start)
        self.runReactor(reactor)

    def test_pauseAndResumeProducing(self):
        """
        Pause producing and then resume producing.
        """

        def pauseAndResume(reactor):
            try:
                protocol = ProcessProtocol()
                transport = reactor.spawnProcess(protocol, pyExe, [pyExe, b'-c', b''], usePTY=self.usePTY)
                transport.pauseProducing()
                transport.resumeProducing()
            finally:
                reactor.stop()
        reactor = self.buildReactor()
        reactor.callWhenRunning(pauseAndResume, reactor)
        self.runReactor(reactor)

    def test_processCommandLineArguments(self):
        """
        Arguments given to spawnProcess are passed to the child process as
        originally intended.
        """
        us = b'twisted.internet.test.process_cli'
        args = [b'hello', b'"', b' \t|<>^&', b'"\\\\"hello\\\\"', b'"foo\\ bar baz\\""']
        allChars = ''.join(map(chr, range(1, 255)))
        if isinstance(allChars, str):
            allChars.encode('utf-8')
        reactor = self.buildReactor()

        def processFinished(finishedArgs):
            output, err, code = finishedArgs
            output = output.split(b'\x00')
            output.pop()
            self.assertEqual(args, output)

        def shutdown(result):
            reactor.stop()
            return result

        def spawnChild():
            d = succeed(None)
            d.addCallback(lambda dummy: utils.getProcessOutputAndValue(pyExe, [b'-m', us] + args, env=properEnv, reactor=reactor))
            d.addCallback(processFinished)
            d.addBoth(shutdown)
        reactor.callWhenRunning(spawnChild)
        self.runReactor(reactor)

    @onlyOnPOSIX
    def test_process_unregistered_before_protocol_ended_callback(self):
        """
        Process is removed from reapProcessHandler dict before running
        ProcessProtocol.processEnded() callback.
        """
        results = []

        class TestProcessProtocol(ProcessProtocol):
            """
            Process protocol captures own presence in
            process.reapProcessHandlers at time of .processEnded() callback.

            @ivar deferred: A deferred fired when the .processEnded() callback
                has completed.
            @type deferred: L{Deferred<defer.Deferred>}
            """

            def __init__(self):
                self.deferred = Deferred()

            def processEnded(self, status):
                """
                Capture whether the process has already been removed
                from process.reapProcessHandlers.

                @param status: unused
                """
                from twisted.internet import process
                handlers = process.reapProcessHandlers
                processes = handlers.values()
                if self.transport in processes:
                    results.append('process present but should not be')
                else:
                    results.append('process already removed as desired')
                self.deferred.callback(None)

        @inlineCallbacks
        def launchProcessAndWait(reactor):
            """
            Launch and wait for a subprocess and allow the TestProcessProtocol
            to capture the order of the .processEnded() callback vs. removal
            from process.reapProcessHandlers.

            @param reactor: Reactor used to spawn the test process and to be
                stopped when checks are complete.
            @type reactor: object providing
                L{twisted.internet.interfaces.IReactorProcess} and
                L{twisted.internet.interfaces.IReactorCore}.
            """
            try:
                testProcessProtocol = TestProcessProtocol()
                reactor.spawnProcess(testProcessProtocol, pyExe, [pyExe, '--version'])
                yield testProcessProtocol.deferred
            except Exception as e:
                results.append(e)
            finally:
                reactor.stop()
        reactor = self.buildReactor()
        reactor.callWhenRunning(launchProcessAndWait, reactor)
        self.runReactor(reactor)
        hamcrest.assert_that(results, hamcrest.equal_to(['process already removed as desired']))

    def checkSpawnProcessEnvironment(self, spawnKwargs, expectedEnv, usePosixSpawnp):
        """
        Shared code for testing the environment variables
        present in the spawned process.

        The spawned process serializes its environ to stderr or stdout (depending on usePTY)
        which is checked against os.environ of the calling process.
        """
        p = Accumulator()
        d = p.endedDeferred = Deferred()
        reactor = self.buildReactor()
        reactor._neverUseSpawn = not usePosixSpawnp
        reactor.callWhenRunning(reactor.spawnProcess, p, pyExe, [pyExe, b'-c', networkString('import os, sys; env = dict(os.environ); env.pop("LC_CTYPE", None); env.pop("__CF_USER_TEXT_ENCODING", None); sys.stderr.write(str(sorted(env.items())))')], usePTY=self.usePTY, **spawnKwargs)

        def shutdown(ign):
            reactor.stop()
        d.addBoth(shutdown)
        self.runReactor(reactor)
        expectedEnv.pop('LC_CTYPE', None)
        expectedEnv.pop('__CF_USER_TEXT_ENCODING', None)
        self.assertEqual(bytes(str(sorted(expectedEnv.items())), 'utf-8'), p.outF.getvalue() if self.usePTY else p.errF.getvalue())

    def checkSpawnProcessEnvironmentWithPosixSpawnp(self, spawnKwargs, expectedEnv):
        return self.checkSpawnProcessEnvironment(spawnKwargs, expectedEnv, usePosixSpawnp=True)

    def checkSpawnProcessEnvironmentWithFork(self, spawnKwargs, expectedEnv):
        return self.checkSpawnProcessEnvironment(spawnKwargs, expectedEnv, usePosixSpawnp=False)

    @onlyOnPOSIX
    def test_environmentPosixSpawnpEnvNotSet(self):
        """
        An empty environment is passed to the spawned process, when the default value of the C{env}
        is used. That is, when the C{env} argument is not explicitly set.

        In this case posix_spawnp is used as the backend for spawning processes.
        """
        return self.checkSpawnProcessEnvironmentWithPosixSpawnp({}, {})

    @onlyOnPOSIX
    def test_environmentForkEnvNotSet(self):
        """
        An empty environment is passed to the spawned process, when the default value of the C{env}
        is used. That is, when the C{env} argument is not explicitly set.

        In this case fork+execvpe is used as the backend for spawning processes.
        """
        return self.checkSpawnProcessEnvironmentWithFork({}, {})

    @onlyOnPOSIX
    def test_environmentPosixSpawnpEnvNone(self):
        """
        The parent process environment is passed to the spawned process, when C{env} is set to
        C{None}.

        In this case posix_spawnp is used as the backend for spawning processes.
        """
        return self.checkSpawnProcessEnvironmentWithPosixSpawnp({'env': None}, os.environ)

    @onlyOnPOSIX
    def test_environmentForkEnvNone(self):
        """
        The parent process environment is passed to the spawned process, when C{env} is set to
        C{None}.

        In this case fork+execvpe is used as the backend for spawning processes.
        """
        return self.checkSpawnProcessEnvironmentWithFork({'env': None}, os.environ)

    @onlyOnPOSIX
    def test_environmentPosixSpawnpEnvCustom(self):
        """
        The user-specified environment without extra variables from parent process is passed to the
        spawned process, when C{env} is set to a dictionary.

        In this case posix_spawnp is used as the backend for spawning processes.
        """
        return self.checkSpawnProcessEnvironmentWithPosixSpawnp({'env': {'MYENV1': 'myvalue1'}}, {'MYENV1': 'myvalue1'})

    @onlyOnPOSIX
    def test_environmentForkEnvCustom(self):
        """
        The user-specified environment without extra variables from parent process is passed to the
        spawned process, when C{env} is set to a dictionary.

        In this case fork+execvpe is used as the backend for spawning processes.
        """
        return self.checkSpawnProcessEnvironmentWithFork({'env': {'MYENV1': 'myvalue1'}}, {'MYENV1': 'myvalue1'})