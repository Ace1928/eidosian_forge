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
class ConstructorTestsMixin:
    """
    Helper methods for verifying default attribute values and corresponding
    constructor arguments.
    """

    def _verifyConstructorArgument(self, argName, defaultVal, altVal):
        """
        Wrap L{verifyConstructorArgument} to provide simpler interface for
        testing Message and _EDNSMessage constructor arguments.

        @param argName: The name of the constructor argument.
        @param defaultVal: The expected default value.
        @param altVal: An alternative value which is expected to be assigned to
            a correspondingly named attribute.
        """
        verifyConstructorArgument(testCase=self, cls=self.messageFactory, argName=argName, defaultVal=defaultVal, altVal=altVal)

    def _verifyConstructorFlag(self, argName, defaultVal):
        """
        Wrap L{verifyConstructorArgument} to provide simpler interface for
        testing  _EDNSMessage constructor flags.

        @param argName: The name of the constructor flag argument
        @param defaultVal: The expected default value of the flag
        """
        assert defaultVal in (True, False)
        verifyConstructorArgument(testCase=self, cls=self.messageFactory, argName=argName, defaultVal=defaultVal, altVal=not defaultVal)