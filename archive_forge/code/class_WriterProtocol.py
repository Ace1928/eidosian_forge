import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
class WriterProtocol(protocol.Protocol):

    def connectionMade(self):
        self.transport.write(b'Hello Cleveland!\n')
        seq = [b'Goodbye', b' cruel', b' world', b'\n']
        self.transport.writeSequence(seq)
        peer = self.transport.getPeer()
        if peer.type != 'TCP':
            msg(f'getPeer returned non-TCP socket: {peer}')
            self.factory.problem = 1
        us = self.transport.getHost()
        if us.type != 'TCP':
            msg(f'getHost returned non-TCP socket: {us}')
            self.factory.problem = 1
        self.factory.done = 1
        self.transport.loseConnection()