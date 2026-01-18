import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class TestController:
    """
    Pretend to be a DNS query processor for a DNSDatagramProtocol.

    @ivar messages: the list of received messages.
    @type messages: C{list} of (msg, protocol, address)
    """

    def __init__(self):
        """
        Initialize the controller: create a list of messages.
        """
        self.messages = []

    def messageReceived(self, msg, proto, addr=None):
        """
        Save the message so that it can be checked during the tests.
        """
        self.messages.append((msg, proto, addr))