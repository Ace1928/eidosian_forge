import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
class LineTester(basic.LineReceiver):
    """
    A line receiver that parses data received and make actions on some tokens.

    @type delimiter: C{bytes}
    @ivar delimiter: character used between received lines.
    @type MAX_LENGTH: C{int}
    @ivar MAX_LENGTH: size of a line when C{lineLengthExceeded} will be called.
    @type clock: L{twisted.internet.task.Clock}
    @ivar clock: clock simulating reactor callLater. Pass it to constructor if
        you want to use the pause/rawpause functionalities.
    """
    delimiter = b'\n'
    MAX_LENGTH = 64

    def __init__(self, clock=None):
        """
        If given, use a clock to make callLater calls.
        """
        self.clock = clock

    def connectionMade(self):
        """
        Create/clean data received on connection.
        """
        self.received = []

    def lineReceived(self, line):
        """
        Receive line and make some action for some tokens: pause, rawpause,
        stop, len, produce, unproduce.
        """
        self.received.append(line)
        if line == b'':
            self.setRawMode()
        elif line == b'pause':
            self.pauseProducing()
            self.clock.callLater(0, self.resumeProducing)
        elif line == b'rawpause':
            self.pauseProducing()
            self.setRawMode()
            self.received.append(b'')
            self.clock.callLater(0, self.resumeProducing)
        elif line == b'stop':
            self.stopProducing()
        elif line[:4] == b'len ':
            self.length = int(line[4:])
        elif line.startswith(b'produce'):
            self.transport.registerProducer(self, False)
        elif line.startswith(b'unproduce'):
            self.transport.unregisterProducer()

    def rawDataReceived(self, data):
        """
        Read raw data, until the quantity specified by a previous 'len' line is
        reached.
        """
        data, rest = (data[:self.length], data[self.length:])
        self.length = self.length - len(data)
        self.received[-1] = self.received[-1] + data
        if self.length == 0:
            self.setLineMode(rest)

    def lineLengthExceeded(self, line):
        """
        Adjust line mode when long lines received.
        """
        if len(line) > self.MAX_LENGTH + 1:
            self.setLineMode(line[self.MAX_LENGTH + 1:])