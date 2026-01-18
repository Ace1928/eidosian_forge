import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def _ourServerOurClientTest(self, name=b'session', **kwargs):
    """
        Create a connected SSH client and server protocol pair and return a
        L{Deferred} which fires with an L{SSHTestChannel} instance connected to
        a channel on that SSH connection.
        """
    result = defer.Deferred()
    self.realm = ConchTestRealm(b'testuser')
    p = portal.Portal(self.realm)
    sshpc = ConchTestSSHChecker()
    sshpc.registerChecker(ConchTestPasswordChecker())
    sshpc.registerChecker(conchTestPublicKeyChecker())
    p.registerChecker(sshpc)
    fac = ConchTestServerFactory()
    fac.portal = p
    fac.startFactory()
    self.server = fac.buildProtocol(None)
    self.clientTransport = LoopbackRelay(self.server)
    self.client = ConchTestClient(lambda conn: SSHTestChannel(name, result, conn=conn, **kwargs))
    self.serverTransport = LoopbackRelay(self.client)
    self.server.makeConnection(self.serverTransport)
    self.client.makeConnection(self.clientTransport)
    return result