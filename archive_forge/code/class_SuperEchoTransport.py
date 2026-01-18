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
class SuperEchoTransport:

    def __init__(self, p):
        self.proto = p
        p.makeConnection(self)
        self.closed = 0

    def write(self, data):
        self.proto.outReceived(data)
        self.proto.outReceived(b'\r\n')
        self.proto.errReceived(data)
        self.proto.errReceived(b'\r\n')

    def loseConnection(self):
        if self.closed:
            return
        self.closed = 1
        self.proto.inConnectionLost()
        self.proto.outConnectionLost()
        self.proto.errConnectionLost()
        self.proto.processEnded(failure.Failure(ProcessTerminated(0, None, None)))