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
def _writeTest(self, write):
    """
        Helper for testing L{IProcessTransport} write functionality.  This
        method spawns a child process and gives C{write} a chance to write some
        bytes to it.  It then verifies that the bytes were actually written to
        it (by relying on the child process to echo them back).

        @param write: A two-argument callable.  This is invoked with a process
            transport and some bytes to write to it.
        """
    reactor = self.buildReactor()
    ended = Deferred()
    protocol = _ShutdownCallbackProcessProtocol(ended)
    bytesToSend = b'hello, world' + networkString(os.linesep)
    program = b'import sys\nsys.stdout.write(sys.stdin.readline())\n'

    def startup():
        transport = reactor.spawnProcess(protocol, pyExe, [pyExe, b'-c', program])
        try:
            write(transport, bytesToSend)
        except BaseException:
            err(None, 'Unhandled exception while writing')
            transport.signalProcess('KILL')
    reactor.callWhenRunning(startup)
    ended.addCallback(lambda ignored: reactor.stop())
    self.runReactor(reactor)
    self.assertEqual(bytesToSend, b''.join(protocol.received[1]))