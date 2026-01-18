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
class ConchSessionForTestAvatar:
    """
    An ISession adapter for ConchTestAvatar.
    """

    def __init__(self, avatar):
        """
        Initialize the session and create a reference to it on the avatar for
        later inspection.
        """
        self.avatar = avatar
        self.avatar._testSession = self
        self.cmd = None
        self.proto = None
        self.ptyReq = False
        self.eof = 0
        self.onClose = defer.Deferred()

    def getPty(self, term, windowSize, attrs):
        log.msg('pty req')
        self._terminalType = term
        self._windowSize = windowSize
        self.ptyReq = True

    def openShell(self, proto):
        log.msg('opening shell')
        self.proto = proto
        EchoTransport(proto)
        self.cmd = b'shell'

    def execCommand(self, proto, cmd):
        self.cmd = cmd
        self.proto = proto
        f = cmd.split()[0]
        if f == b'false':
            t = FalseTransport(proto)
            reactor.callLater(0, t.loseConnection)
        elif f == b'echo':
            t = EchoTransport(proto)
            t.write(cmd[5:])
            t.loseConnection()
        elif f == b'secho':
            t = SuperEchoTransport(proto)
            t.write(cmd[6:])
            t.loseConnection()
        elif f == b'eecho':
            t = ErrEchoTransport(proto)
            t.write(cmd[6:])
            t.loseConnection()
        else:
            raise error.ConchError('bad exec')
        self.avatar.conn.transport.expectedLoseConnection = 1

    def eofReceived(self):
        self.eof = 1

    def closed(self):
        log.msg('closed cmd "%s"' % self.cmd)
        self.remoteWindowLeftAtClose = self.proto.session.remoteWindowLeft
        self.onClose.callback(None)