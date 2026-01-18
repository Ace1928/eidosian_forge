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
class AmpBoxTests(TestCase):
    """
    Test a few essential properties of AMP boxes, mostly with respect to
    serialization correctness.
    """

    def test_serializeStr(self):
        """
        Make sure that strs serialize to strs.
        """
        a = amp.AmpBox(key=b'value')
        self.assertEqual(type(a.serialize()), bytes)

    def test_serializeUnicodeKeyRaises(self):
        """
        Verify that TypeError is raised when trying to serialize Unicode keys.
        """
        a = amp.AmpBox(**{'key': 'value'})
        self.assertRaises(TypeError, a.serialize)

    def test_serializeUnicodeValueRaises(self):
        """
        Verify that TypeError is raised when trying to serialize Unicode
        values.
        """
        a = amp.AmpBox(key='value')
        self.assertRaises(TypeError, a.serialize)