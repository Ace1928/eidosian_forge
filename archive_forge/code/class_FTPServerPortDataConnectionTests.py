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
class FTPServerPortDataConnectionTests(FTPServerPasvDataConnectionTests):

    def setUp(self):
        self.dataPorts = []
        return FTPServerPasvDataConnectionTests.setUp(self)

    def _makeDataConnection(self, ignored=None):
        deferred = defer.Deferred()

        class DataFactory(protocol.ServerFactory):
            protocol = _BufferingProtocol

            def buildProtocol(self, addr):
                p = protocol.ServerFactory.buildProtocol(self, addr)
                reactor.callLater(0, deferred.callback, p)
                return p
        dataPort = reactor.listenTCP(0, DataFactory(), interface='127.0.0.1')
        self.dataPorts.append(dataPort)
        cmd = 'PORT ' + ftp.encodeHostPort('127.0.0.1', dataPort.getHost().port)
        self.client.queueStringCommand(cmd)
        return deferred

    def tearDown(self):
        """
        Tear down the connection.

        @return: L{defer.DeferredList}
        """
        l = [defer.maybeDeferred(port.stopListening) for port in self.dataPorts]
        d = defer.maybeDeferred(FTPServerPasvDataConnectionTests.tearDown, self)
        l.append(d)
        return defer.DeferredList(l, fireOnOneErrback=True)

    def test_PORTCannotConnect(self):
        """
        Listen on a port, and immediately stop listening as a way to find a
        port number that is definitely closed.
        """
        d = self._anonymousLogin()

        def loggedIn(ignored):
            port = reactor.listenTCP(0, protocol.Factory(), interface='127.0.0.1')
            portNum = port.getHost().port
            d = port.stopListening()
            d.addCallback(lambda _: portNum)
            return d
        d.addCallback(loggedIn)

        def gotPortNum(portNum):
            return self.assertCommandFailed('PORT ' + ftp.encodeHostPort('127.0.0.1', portNum), ["425 Can't open data connection."])
        return d.addCallback(gotPortNum)

    def test_nlstGlobbing(self):
        """
        When Unix shell globbing is used with NLST only files matching the
        pattern will be returned.
        """
        self.dirPath.child('test.txt').touch()
        self.dirPath.child('ceva.txt').touch()
        self.dirPath.child('no.match').touch()
        d = self._anonymousLogin()
        self._download('NLST *.txt', chainDeferred=d)

        def checkDownload(download):
            filenames = download[:-2].split(b'\r\n')
            filenames.sort()
            self.assertEqual([b'ceva.txt', b'test.txt'], filenames)
        return d.addCallback(checkDownload)