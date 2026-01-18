import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class FTPServerTestCase(TestCase):
    """
    Simple tests for an FTP server with the default settings.

    @ivar clientFactory: class used as ftp client.
    """
    clientFactory = ftp.FTPClientBasic
    userAnonymous = 'anonymous'

    def setUp(self):
        protocols = []
        self.directory = self.mktemp()
        os.mkdir(self.directory)
        self.dirPath = filepath.FilePath(self.directory)
        p = portal.Portal(ftp.FTPRealm(anonymousRoot=self.directory, userHome=self.directory))
        p.registerChecker(checkers.AllowAnonymousAccess(), credentials.IAnonymous)
        users_checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.username = 'test-user'
        self.password = 'test-password'
        users_checker.addUser(self.username, self.password)
        p.registerChecker(users_checker, credentials.IUsernamePassword)
        self.factory = ftp.FTPFactory(portal=p, userAnonymous=self.userAnonymous)
        self.port = port = reactor.listenTCP(0, self.factory, interface='127.0.0.1')
        self.addCleanup(port.stopListening)
        buildProtocol = self.factory.buildProtocol
        d1 = defer.Deferred()

        def _rememberProtocolInstance(addr):
            del self.factory.buildProtocol
            protocol = buildProtocol(addr)
            self.serverProtocol = protocol.wrappedProtocol

            def cleanupServer():
                if self.serverProtocol.transport is not None:
                    self.serverProtocol.transport.loseConnection()
            self.addCleanup(cleanupServer)
            d1.callback(None)
            protocols.append(protocol)
            return protocol
        self.factory.buildProtocol = _rememberProtocolInstance
        portNum = port.getHost().port
        clientCreator = protocol.ClientCreator(reactor, self.clientFactory)
        d2 = clientCreator.connectTCP('127.0.0.1', portNum)

        def gotClient(client):
            self.client = client
            self.addCleanup(self.client.transport.loseConnection)
            protocols.append(self.client)
        d2.addCallback(gotClient)
        self.addCleanup(proto_helpers.waitUntilAllDisconnected, reactor, protocols)
        return defer.gatherResults([d1, d2])

    def assertCommandResponse(self, command, expectedResponseLines, chainDeferred=None):
        """
        Asserts that a sending an FTP command receives the expected
        response.

        Returns a Deferred.  Optionally accepts a deferred to chain its actions
        to.
        """
        if chainDeferred is None:
            chainDeferred = defer.succeed(None)

        def queueCommand(ignored):
            d = self.client.queueStringCommand(command)

            def gotResponse(responseLines):
                self.assertEqual(expectedResponseLines, responseLines)
            return d.addCallback(gotResponse)
        return chainDeferred.addCallback(queueCommand)

    def assertCommandFailed(self, command, expectedResponse=None, chainDeferred=None):
        if chainDeferred is None:
            chainDeferred = defer.succeed(None)

        def queueCommand(ignored):
            return self.client.queueStringCommand(command)
        chainDeferred.addCallback(queueCommand)
        self.assertFailure(chainDeferred, ftp.CommandFailed)

        def failed(exception):
            if expectedResponse is not None:
                self.assertEqual(expectedResponse, exception.args[0])
        return chainDeferred.addCallback(failed)

    def _anonymousLogin(self):
        d = self.assertCommandResponse('USER anonymous', ['331 Guest login ok, type your email address as password.'])
        return self.assertCommandResponse('PASS test@twistedmatrix.com', ['230 Anonymous login ok, access restrictions apply.'], chainDeferred=d)

    def _userLogin(self):
        """
        Authenticates the FTP client using the test account.

        @return: L{Deferred} of command response
        """
        d = self.assertCommandResponse('USER %s' % self.username, ['331 Password required for %s.' % self.username])
        return self.assertCommandResponse('PASS %s' % self.password, ['230 User logged in, proceed'], chainDeferred=d)