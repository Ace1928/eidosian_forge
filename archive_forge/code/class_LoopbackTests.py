from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class LoopbackTests(unittest.TestCase):
    if keys is None:
        skip = 'cannot run without cryptography'

    class Factory:

        class Service:
            name = b'TestService'

            def serviceStarted(self):
                self.transport.loseConnection()

            def serviceStopped(self):
                pass

        def getService(self, avatar, name):
            return self.Service

    def test_loopback(self):
        """
        Test that the userauth server and client play nicely with each other.
        """
        server = userauth.SSHUserAuthServer()
        client = ClientUserAuth(b'foo', self.Factory.Service())
        server.transport = transport.SSHTransportBase()
        server.transport.service = server
        server.transport.isEncrypted = lambda x: True
        client.transport = transport.SSHTransportBase()
        client.transport.service = client
        server.transport.sessionID = client.transport.sessionID = b''
        server.transport.sendKexInit = client.transport.sendKexInit = lambda: None
        server.transport.factory = self.Factory()
        server.passwordDelay = 0
        realm = Realm()
        portal = Portal(realm)
        checker = SSHProtocolChecker()
        checker.registerChecker(PasswordChecker())
        checker.registerChecker(PrivateKeyChecker())
        checker.areDone = lambda aId: len(checker.successfulCredentials[aId]) == 2
        portal.registerChecker(checker)
        server.transport.factory.portal = portal
        d = loopback.loopbackAsync(server.transport, client.transport)
        server.transport.transport.logPrefix = lambda: '_ServerLoopback'
        client.transport.transport.logPrefix = lambda: '_ClientLoopback'
        server.serviceStarted()
        client.serviceStarted()

        def check(ignored):
            self.assertEqual(server.transport.service.name, b'TestService')
        return d.addCallback(check)