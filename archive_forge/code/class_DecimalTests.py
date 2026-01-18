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
class DecimalTests(TestCase):
    """
    Tests for L{amp.Decimal}.
    """

    def test_nonDecimal(self):
        """
        L{amp.Decimal.toString} raises L{ValueError} if passed an object which
        is not an instance of C{decimal.Decimal}.
        """
        argument = amp.Decimal()
        self.assertRaises(ValueError, argument.toString, '1.234')
        self.assertRaises(ValueError, argument.toString, 1.234)
        self.assertRaises(ValueError, argument.toString, 1234)