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
class FTPClientFailedRETRAndErrbacksUponDisconnectTests(TestCase):
    """
    FTP client fails and RETR fails and disconnects.
    """

    def test_FailedRETR(self):
        """
        RETR fails.
        """
        f = protocol.Factory()
        f.noisy = 0
        port = reactor.listenTCP(0, f, interface='127.0.0.1')
        self.addCleanup(port.stopListening)
        portNum = port.getHost().port
        responses = ['220 ready, dude (vsFTPd 1.0.0: beat me, break me)', '331 Please specify the password.', '230 Login successful. Have fun.', '200 Binary it is, then.', '227 Entering Passive Mode (127,0,0,1,%d,%d)' % (portNum >> 8, portNum & 255), '550 Failed to open file.']
        f.buildProtocol = lambda addr: PrintLines(responses)
        cc = protocol.ClientCreator(reactor, ftp.FTPClient, passive=1)
        d = cc.connectTCP('127.0.0.1', portNum)

        def gotClient(client):
            p = protocol.Protocol()
            return client.retrieveFile('/file/that/doesnt/exist', p)
        d.addCallback(gotClient)
        return self.assertFailure(d, ftp.CommandFailed)

    def test_errbacksUponDisconnect(self):
        """
        Test the ftp command errbacks when a connection lost happens during
        the operation.
        """
        ftpClient = ftp.FTPClient()
        tr = proto_helpers.StringTransportWithDisconnection()
        ftpClient.makeConnection(tr)
        tr.protocol = ftpClient
        d = ftpClient.list('some path', Dummy())
        m = []

        def _eb(failure):
            m.append(failure)
            return None
        d.addErrback(_eb)
        from twisted.internet.main import CONNECTION_LOST
        ftpClient.connectionLost(failure.Failure(CONNECTION_LOST))
        self.assertTrue(m, m)
        return d