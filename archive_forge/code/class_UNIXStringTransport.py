import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
@implementer(interfaces.IUNIXTransport)
class UNIXStringTransport:
    """
    An in-memory implementation of L{interfaces.IUNIXTransport} which collects
    all data given to it for later inspection.

    @ivar _queue: A C{list} of the data which has been given to this transport,
        eg via C{write} or C{sendFileDescriptor}.  Elements are two-tuples of a
        string (identifying the destination of the data) and the data itself.
    """

    def __init__(self, descriptorFuzz):
        """
        @param descriptorFuzz: An offset to apply to descriptors.
        @type descriptorFuzz: C{int}
        """
        self._fuzz = descriptorFuzz
        self._queue = []

    def sendFileDescriptor(self, descriptor):
        self._queue.append(('fileDescriptorReceived', descriptor + self._fuzz))

    def write(self, data):
        self._queue.append(('dataReceived', data))

    def writeSequence(self, seq):
        for data in seq:
            self.write(data)

    def loseConnection(self):
        self._queue.append(('connectionLost', Failure(error.ConnectionLost())))

    def getHost(self):
        return address.UNIXAddress('/tmp/some-path')

    def getPeer(self):
        return address.UNIXAddress('/tmp/another-path')