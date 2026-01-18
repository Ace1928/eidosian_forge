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
class ConchTestClient(ConchTestBase, transport.SSHClientTransport):
    """
        @ivar _channelFactory: A callable which accepts an SSH connection and
            returns a channel which will be attached to a new channel on that
            connection.
        """

    def __init__(self, channelFactory):
        self._channelFactory = channelFactory

    def connectionLost(self, reason):
        ConchTestBase.connectionLost(self, reason)
        transport.SSHClientTransport.connectionLost(self, reason)

    def verifyHostKey(self, key, fp):
        keyMatch = key == keys.Key.fromString(publicRSA_openssh).blob()
        fingerprintMatch = fp == b'85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da'
        if keyMatch and fingerprintMatch:
            return defer.succeed(1)
        return defer.fail(Exception('Key or fingerprint mismatch'))

    def connectionSecure(self):
        self.requestService(ConchTestClientAuth(b'testuser', ConchTestClientConnection(self._channelFactory)))