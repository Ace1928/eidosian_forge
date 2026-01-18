import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
class CFTPClientTestBase(SFTPTestBase):

    def setUp(self):
        with open('dsa_test.pub', 'wb') as f:
            f.write(test_ssh.publicDSA_openssh)
        with open('dsa_test', 'wb') as f:
            f.write(test_ssh.privateDSA_openssh)
        os.chmod('dsa_test', 33152)
        with open('kh_test', 'wb') as f:
            f.write(b'127.0.0.1 ' + test_ssh.publicRSA_openssh)
        return SFTPTestBase.setUp(self)

    def startServer(self):
        realm = FileTransferTestRealm(self.testDir)
        p = portal.Portal(realm)
        p.registerChecker(test_ssh.conchTestPublicKeyChecker())
        fac = test_ssh.ConchTestServerFactory()
        fac.portal = p
        self.server = reactor.listenTCP(0, fac, interface='127.0.0.1')

    def stopServer(self):
        if not hasattr(self.server.factory, 'proto'):
            return self._cbStopServer(None)
        self.server.factory.proto.expectedLoseConnection = 1
        d = defer.maybeDeferred(self.server.factory.proto.transport.loseConnection)
        d.addCallback(self._cbStopServer)
        return d

    def _cbStopServer(self, ignored):
        return defer.maybeDeferred(self.server.stopListening)

    def tearDown(self):
        for f in ['dsa_test.pub', 'dsa_test', 'kh_test']:
            try:
                os.remove(f)
            except BaseException:
                pass
        return SFTPTestBase.tearDown(self)