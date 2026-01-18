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
class LineOnlyTester(basic.LineOnlyReceiver):
    """
    A buffering line only receiver.
    """
    delimiter = b'\n'
    MAX_LENGTH = 64

    def connectionMade(self):
        """
        Create/clean data received on connection.
        """
        self.received = []

    def lineReceived(self, line):
        """
        Save received data.
        """
        self.received.append(line)