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
class FlippingLineTester(basic.LineReceiver):
    """
    A line receiver that flips between line and raw data modes after one byte.
    """
    delimiter = b'\n'

    def __init__(self):
        self.lines = []

    def lineReceived(self, line):
        """
        Set the mode to raw.
        """
        self.lines.append(line)
        self.setRawMode()

    def rawDataReceived(self, data):
        """
        Set the mode back to line.
        """
        self.setLineMode(data[1:])