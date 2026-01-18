import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
class _DummyTransport:

    def __init__(self, proto):
        self.proto = proto

    def dataReceived(self, data):
        self.proto.transport.write(data)

    def write(self, data):
        self.proto.dataReceived(data)

    def writeSequence(self, seq):
        self.write(b''.join(seq))

    def loseConnection(self):
        self.proto.connectionLost(protocol.connectionDone)