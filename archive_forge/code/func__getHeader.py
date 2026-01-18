from struct import calcsize, pack, unpack
from twisted.protocols.stateful import StatefulProtocol
from twisted.protocols.test import test_basic
from twisted.trial.unittest import TestCase
def _getHeader(self, msg):
    length, = unpack('!i', msg)
    if length > self.MAX_LENGTH:
        self.lengthLimitExceeded(length)
        return
    return (self._getString, length)