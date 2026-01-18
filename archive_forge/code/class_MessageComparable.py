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
class MessageComparable(FancyEqMixin, FancyStrMixin):
    """
    A wrapper around L{dns.Message} which is comparable so that it can be tested
    using some of the L{dns._EDNSMessage} tests.
    """
    showAttributes = compareAttributes = ('id', 'answer', 'opCode', 'auth', 'trunc', 'recDes', 'recAv', 'rCode', 'queries', 'answers', 'authority', 'additional')

    def __init__(self, original):
        self.original = original

    def __getattr__(self, key):
        return getattr(self.original, key)