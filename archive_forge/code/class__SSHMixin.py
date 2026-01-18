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
class _SSHMixin(_BaseMixin):

    def setUp(self):
        if not ssh:
            raise SkipTest("cryptography requirements missing, can't run historic recvline tests over ssh")
        u, p = (b'testuser', b'testpass')
        rlm = TerminalRealm()
        rlm.userFactory = TestUser
        rlm.chainedProtocolFactory = lambda: insultsServer
        checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        checker.addUser(u, p)
        ptl = portal.Portal(rlm)
        ptl.registerChecker(checker)
        sshFactory = ConchFactory(ptl)
        sshKey = keys._getPersistentRSAKey(filepath.FilePath(self.mktemp()), keySize=1024)
        sshFactory.publicKeys[b'ssh-rsa'] = sshKey
        sshFactory.privateKeys[b'ssh-rsa'] = sshKey
        sshFactory.serverProtocol = self.serverProtocol
        sshFactory.startFactory()
        recvlineServer = self.serverProtocol()
        insultsServer = insults.ServerProtocol(lambda: recvlineServer)
        sshServer = sshFactory.buildProtocol(None)
        clientTransport = LoopbackRelay(sshServer)
        recvlineClient = NotifyingExpectableBuffer()
        insultsClient = insults.ClientProtocol(lambda: recvlineClient)
        sshClient = TestTransport(lambda: insultsClient, (), {}, u, p, self.WIDTH, self.HEIGHT)
        serverTransport = LoopbackRelay(sshClient)
        sshClient.makeConnection(clientTransport)
        sshServer.makeConnection(serverTransport)
        self.recvlineClient = recvlineClient
        self.sshClient = sshClient
        self.sshServer = sshServer
        self.clientTransport = clientTransport
        self.serverTransport = serverTransport
        return recvlineClient.onConnection

    def _testwrite(self, data):
        self.sshClient.write(data)