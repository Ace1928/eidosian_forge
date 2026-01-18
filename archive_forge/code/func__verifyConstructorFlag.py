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
def _verifyConstructorFlag(self, argName, defaultVal):
    """
        Wrap L{verifyConstructorArgument} to provide simpler interface for
        testing  _EDNSMessage constructor flags.

        @param argName: The name of the constructor flag argument
        @param defaultVal: The expected default value of the flag
        """
    assert defaultVal in (True, False)
    verifyConstructorArgument(testCase=self, cls=self.messageFactory, argName=argName, defaultVal=defaultVal, altVal=not defaultVal)