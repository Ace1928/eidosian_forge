import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
class _StdioMixin(_BaseMixin):

    def setUp(self):
        testTerminal = NotifyingExpectableBuffer()
        insultsClient = insults.ClientProtocol(lambda: testTerminal)
        processClient = stdio.TerminalProcessProtocol(insultsClient)
        exe = sys.executable
        module = stdio.__file__
        if module.endswith('.pyc') or module.endswith('.pyo'):
            module = module[:-1]
        args = [exe, module, reflect.qual(self.serverProtocol)]
        from twisted.internet import reactor
        clientTransport = reactor.spawnProcess(processClient, exe, args, env=properEnv, usePTY=True)
        self.recvlineClient = self.testTerminal = testTerminal
        self.processClient = processClient
        self.clientTransport = clientTransport
        return defer.gatherResults(filter(None, [processClient.onConnection, testTerminal.expect(b'>>> ')]))

    def tearDown(self):
        try:
            self.clientTransport.signalProcess('KILL')
        except (error.ProcessExitedAlready, OSError):
            pass

        def trap(failure):
            failure.trap(error.ProcessTerminated)
            self.assertIsNone(failure.value.exitCode)
            self.assertEqual(failure.value.status, 9)
        return self.testTerminal.onDisconnection.addErrback(trap)

    def _testwrite(self, data):
        self.clientTransport.write(data)