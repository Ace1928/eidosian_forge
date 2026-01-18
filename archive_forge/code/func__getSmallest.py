import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def _getSmallest(self):
    bytes = self.enc.prefixLimit
    bits = bytes * 7
    largest = 2 ** bits - 1
    smallest = largest + 1
    return smallest