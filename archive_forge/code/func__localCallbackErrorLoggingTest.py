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
def _localCallbackErrorLoggingTest(self, callResult):
    """
        Verify that C{callResult} completes with a L{None} result and that an
        unhandled error has been logged.
        """
    finalResult = []
    callResult.addBoth(finalResult.append)
    self.assertEqual(1, len(self.sender.unhandledErrors))
    self.assertIsInstance(self.sender.unhandledErrors[0].value, ZeroDivisionError)
    self.assertEqual([None], finalResult)