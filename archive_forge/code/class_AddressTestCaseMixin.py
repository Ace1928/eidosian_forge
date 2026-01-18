import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class AddressTestCaseMixin:

    def test_addressComparison(self):
        """
        Two different address instances, sharing the same properties are
        considered equal by C{==} and not considered not equal by C{!=}.

        Note: When applied via UNIXAddress class, this uses the same
        filename for both objects being compared.
        """
        self.assertTrue(self.buildAddress() == self.buildAddress())
        self.assertFalse(self.buildAddress() != self.buildAddress())

    def test_hash(self):
        """
        C{__hash__} can be used to get a hash of an address, allowing
        addresses to be used as keys in dictionaries, for instance.
        """
        addr = self.buildAddress()
        d = {addr: True}
        self.assertTrue(d[self.buildAddress()])

    def test_differentNamesComparison(self):
        """
        Check that comparison operators work correctly on address objects
        when a different name is passed in
        """
        self.assertFalse(self.buildAddress() == self.buildDifferentAddress())
        self.assertFalse(self.buildDifferentAddress() == self.buildAddress())
        self.assertTrue(self.buildAddress() != self.buildDifferentAddress())
        self.assertTrue(self.buildDifferentAddress() != self.buildAddress())

    def assertDeprecations(self, testMethod, message):
        """
        Assert that the a DeprecationWarning with the given message was
        emitted against the given method.
        """
        warnings = self.flushWarnings([testMethod])
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], message)
        self.assertEqual(len(warnings), 1)